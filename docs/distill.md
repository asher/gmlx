# Distill a teacher into a student

This guide is for anyone who wants a small GGUF model to know something a
larger one knows, without the larger one running at inference. It walks the
`gmlx distill` actions through one worked task, says what each step costs,
and explains the numbers the reports print.

You give it a document, a list of questions about it, a large GGUF (the
teacher, which reads the document) and a small GGUF (the student, which
never sees it). You get a small adapter GGUF that you attach to the student
with `--adapter`, after which the student answers the questions without the
document in its prompt. On the worked task the adapter gets 0.88 of the
held-out questions right, against 0.95 with the document pasted into every
prompt, and costs nothing per request. The same steps train a student on
plain text, or on a behavior such as answering in a fixed format.

- [Before you start](#before-you-start)
- [Write the inputs](#write-the-inputs)
- [Check that the document matters](#check-that-the-document-matters)
- [Round one, the teacher writes the corpus](#round-one-the-teacher-writes-the-corpus)
- [Use and measure the adapter](#use-and-measure-the-adapter)
- [Round two, the student writes and the teacher scores](#round-two-the-student-writes-and-the-teacher-scores)
- [What each step costs](#what-each-step-costs)
- [Read the numbers](#read-the-numbers)
- [What to expect](#what-to-expect)
- [A ten-minute smoke run](#a-ten-minute-smoke-run)
- [Train on plain text instead](#train-on-plain-text-instead)
- [When something goes wrong](#when-something-goes-wrong)
- [Advanced settings](#advanced-settings)
- [Limitations](#limitations)

## Before you start

Pick a teacher and a student from the same model family when you can. The
two then split text into the same word pieces, every position of the
teacher's output is a direct target for the student, and the result in [What
to expect](#what-to-expect) is the same-tokenizer case. A student from
another family still works, through the mapping described under [Advanced
settings](#advanced-settings), but reaches a fraction of the same result.

The pipeline runs the teacher on its own, once, and never loads the teacher
and the student together. Memory is set by the larger of the two: the
teacher during `gen` and `cache`, the student during `train`. A 27B teacher
at Q8 and a 9B student at Q6_K fit a 64 GB machine in every step. The
figures are in [What each step costs](#what-each-step-costs).

Six actions do the work and one checks an input. Every flag of every action
is listed under [gmlx distill](cli.md#gmlx-distill) in the CLI reference.

| Action | What it does |
|---|---|
| `gen` | Serves the teacher and writes its replies to your prompts as a corpus. |
| `filter` | Drops replies the student should not learn from, including any your own checker rejects. |
| `cache` | Runs the teacher over the corpus once and stores, per position, which next tokens it favored and by how much. |
| `align` | Reads the cache with the student's tokenizer and writes the training targets, an aligned cache. |
| `train` | Fits a LoRA adapter on the student against one or more aligned caches and writes it as a GGUF. |
| `eval` | Scores the student with and without the adapter on held-out text and tasks. |
| `census` | Measures how much a document moves the teacher, from two caches of the same replies. |

## Write the inputs

The worked task is a database schema. The teacher reads the schema and
answers questions with SQL, the student must answer the same questions with
no schema in its prompt, and a query counts as right when it runs against
the database and returns the reference rows. Any document with a checkable
task fits the same steps: an API reference with tests, a style guide with a
linter, a rulebook with a judge. Four files go in.

Only the teacher reads the document, one text file, `schema.md` below. The
database, `freight.sqlite` below, is what the checker runs queries against
and is not part of the pipeline.

The prompts are a jsonl file with one question per line. Each row has an
`id`, a `messages` list that ends on a user turn, and any extra fields you
want to travel with the row. The row below carries its expected answer under
`check`, which `gen` copies onto the reply row unchanged, so the checker
finds it there:

```json
{"id": "train-00000", "family": "in_transit_hull",
 "messages": [{"role": "user", "content": "For Kestrel-class ships, how many manifests have no arrival yet?\n\nAnswer with one SQLite query in a ```sql code block and nothing else."}],
 "check": {"sql": "SELECT COUNT(*) FROM manifests m JOIN haulers h ON m.hauler_id = h.hauler_id WHERE h.hull_class = 'Kestrel' AND m.arrived_at IS NULL", "ordered": false}}
```

Write three prompt sets. Training prompts cover every kind of question the
document answers, with several hundred rows and at least three phrasings of
each. Held-out prompts ask the same kinds of question in words the training
set does not use, and measure what the student learned. A third set asks
kinds of question the training set never covers, and measures whether the
student learned the document or only the training questions. Include
training prompts that combine two kinds of question in one, such as a count
over a join. Without them the student answers each kind and fails the
combinations. Rows that describe the document in prose rather than using it
do not help and stay out.

The checker is any command that reads the surviving rows as jsonl on stdin
and prints one line per row, `ok` or a reason word. The filter runs it
through the shell, so a script with arguments works. A skeleton for the SQL
task:

```python
#!/usr/bin/env python3
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
for line in sys.stdin:
    row = json.loads(line)
    reply = row["messages"][-1]["content"]
    sql = reply.split("```sql", 1)[-1].split("```", 1)[0].strip()
    try:
        got = db.execute(sql).fetchall()
        want = db.execute(row["check"]["sql"]).fetchall()
        print("ok" if got == want else "wrong_rows")
    except Exception:
        print("bad_sql")
```

Every row the checker rejects is a row that would have taught the student a
wrong answer, so the checker is not optional. The filter records each reason
word in its report and in the rejects file.

## Check that the document matters

Before spending hours on training, measure whether the document changes what
the teacher says. The check generates the teacher's replies to the held-out
prompts with the document in view, then caches those exact replies twice:
once as written, and once with the document removed from the prompt.
`census` compares the two caches:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-heldout.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-ctx.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus heldout-ctx.jsonl --out cache-heldout-ctx/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --max-len 2560
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus heldout-ctx.jsonl --out cache-heldout-bare/ \
    --messages-key student_messages \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --max-len 2560
gmlx distill census --without cache-heldout-bare/ --with cache-heldout-ctx/ \
    --corpus heldout-ctx.jsonl --out census.json --md census.md
```

`--context` puts the document into the last user turn on the teacher's side
only. Each reply row then carries two message lists, `messages` with the
document and `student_messages` without it, and the second `cache` reads the
bare list. Two rows of the census report decide. The distillable effect is
how far the document moves the teacher's next-token choices, averaged over
every position of the replies, in nats. The `positions above 1.0 nats` row
is the share of positions it moves by more than one nat. An effect under
about 0.05 nats, or a share under one percent, means the document changes
little that a student could learn, and the run is not worth its hours. Keep
the census JSON: the evaluation reads it to score the adapter at the
positions the document moved.

## Round one, the teacher writes the corpus

The teacher answers the training prompts with the document in view and the
checker keeps the replies that were right. The teacher is then cached over
the kept replies, the cache is aligned to the student, and the adapter is
trained:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-train.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r1-replies.jsonl
gmlx distill filter --in r1-replies.jsonl --out r1-corpus.jsonl \
    --max-reply-tokens 1180 --verify "./check-sql.py freight.sqlite" \
    --report r1-filter.json --rejects r1-rejects.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r1-corpus.jsonl --out cache-r1/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r1/ --student Qwen3.5-9B-Q6_K.gguf --out view-r1/
gmlx distill train --view view-r1/ --student Qwen3.5-9B-Q6_K.gguf --adapter-out r1.gguf \
    --iters 302 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5
```

`gen` serves the teacher on a local port, sends `--concurrency` requests at
a time, and stops the server when it is done. `--thinking` turns the
teacher's reasoning on, so each reply carries its reasoning under
`reasoning_content` and the student learns the reasoning as well as the
answer. `--thinking-budget` caps the reasoning at 1000 tokens and
`--max-tokens` gives 320 more for the answer. A run that stops part way
resumes when rerun, since prompt ids already in the output are skipped.
`--temperature` and `--top-p` are the sampling settings, and 0.6 with 0.95
gives varied replies that stay on task.

`filter` runs its checks in a fixed order and names the first one a row
fails. A reply whose reasoning hit the budget is unfinished and is dropped,
so expect to lose about a third of the replies at a 1000-token budget, and
write more prompts than the rows you need. `--max-reply-tokens` drops
replies longer than the cache window below will hold. The report file has
the kept and dropped counts and the rejects file has one reason per dropped
row.

`cache` runs the teacher over the kept replies. `--frame reply-think` tells
it the rows are conversations and the target starts at the final turn's
reasoning. `--frame-kwargs` sets the same thinking switch `gen` used, so the
teacher reads the conversation the way it wrote it. `--max-len` is the
longest row in teacher tokens, and 2560 holds a 1180-token reply behind a
prompt. `--top-k` is how many next-token candidates are stored per position,
and 256 is the default.

`align` runs on the CPU with the two tokenizers only and writes the aligned
cache into `view-r1/`. On a same-family pair it records that the tokenizers
match and finishes in seconds. `train` reads it, fits the adapter on the
quantized student, and writes the adapter GGUF. `--iters` counts steps of
`--batch-size` rows each, and two passes over the rows is enough for a
generated corpus. `--lora-rank` is the adapter's capacity, `--lora-alpha`
its scale, and `--lr` the peak learning rate. The values above are the ones
that worked on the 9B student, and [Advanced settings](#advanced-settings)
says what each one changes.

## Use and measure the adapter

The adapter attaches to the same student GGUF it was trained on. It is tied
to that file, and a different quantization of the same model is a different
student:

```sh
gmlx serve Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf
```

A student trained with `--frame reply-think` learned to reason before
answering, so serve it with thinking on, which is the Qwen default. The
single-model `serve` form registers both the adapted model and the bare
base, as [Use the adapter](lora.md#use-the-adapter) in the LoRA guide
describes.

The pass rate is measured the way the corpus was filtered. Generate the
student's replies to the held-out prompts, without the document, and run the
checker over them:

```sh
gmlx distill gen --teacher Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-heldout.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-r1.jsonl
gmlx distill filter --in heldout-r1.jsonl --out heldout-r1-ok.jsonl \
    --verify "./check-sql.py freight.sqlite" --report heldout-r1.json
```

`--serve-arg` passes an argument through to `gmlx serve`, and two of them
attach the adapter. The kept count in the report over the row count is the
pass rate. Run the same two commands on the untouched student, without the
serve arguments, for the before figure, and on the prompts of the untrained
kinds for the third.

`eval` scores what a pass rate cannot see. With the census JSON from the
check above it scores the adapter at the positions the document moved, and
it checks that the student's general behavior survived:

```sh
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --before \
    --reply-slice heldout=heldout-ctx.jsonl --reply-think --reply-positions census.json \
    --chat-max-len 1536 --md eval.md --json eval.json
```

`--before` scores the same loaded model a second time with the adapter
switched off, so both figures come from one process. `--reply-slice` scores
the teacher's held-out replies from the census check, `--reply-think`
includes their reasoning, and `--reply-positions` keeps only the positions
the document moved. `--chat-max-len` is the longest conversation scored.
[Read the numbers](#read-the-numbers) explains the report.

## Round two, the student writes and the teacher scores

A second round trains the student on the teacher's corrections of its own
replies, and lifts the pass rate a few points over round one. The student
with its first adapter answers the training prompts without the document and
the checker keeps the right replies. The filter then puts the document back
on the teacher's side before the teacher is cached over them:

```sh
gmlx distill gen --teacher Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-train.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r2-replies.jsonl
gmlx distill filter --in r2-replies.jsonl --out r2-corpus.jsonl --max-reply-tokens 1180 \
    --verify "./check-sql.py freight.sqlite" --context schema.md
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r2-corpus.jsonl --out cache-r2/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r2/ --student Qwen3.5-9B-Q6_K.gguf --out view-r2/ --tables view-r1/
gmlx distill train --view view-r1/ --view view-r2/ --student Qwen3.5-9B-Q6_K.gguf \
    --adapter-out r2.gguf --iters 678 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5
```

`filter --context` rebuilds every kept row with the document in the
teacher's prompt and the bare prompt under `student_messages`, so the cache
holds what the teacher thinks of the student's own words with the document
in view, while the student trains on the prompt alone. `align --tables`
reuses the tokenizer tables from round one. `train` takes both aligned
caches and starts from the base weights, not from the first adapter, so
`--iters` doubles with the rows. Measure `r2.gguf` the same way as round
one.

## What each step costs

The figures below are from the worked task on an M5 Max: a 27B teacher at
Q8, a 9B student at Q6_K, 615 training questions plus 264 combined ones, and
two rounds. The whole run took about 11 GPU hours.

| Step | Memory | Time | Disk |
|---|---|---|---|
| `gen` | the teacher, served | decode bound, so `--concurrency` 8 keeps it busy | the replies, a few MB |
| `cache` | the teacher plus a few GB | about 500 teacher tokens per second | 1.6 KB per position at top-k 256 |
| `align` | the tokenizers only | about 2 ms per row on the CPU | a few MB unless `--materialize` |
| `train` | 50.7 GB peak on the 9B student | 9.3 s per step of 3 rows, so 678 steps in 1.75 h | two checkpoints under `--ckpt-dir` |
| `eval` | the student | minutes per slice, longer with the adapter attached | two report files |

A cache is 6 x K + 22 bytes per position plus the text, so a corpus of 600
rows near 1300 tokens takes about 1.2 GB at the default K. `--max-disk-gb`
refuses a cache whose estimate exceeds it, and every size flag counts
decimal GB. Training memory scales with the row length and the batch.
`--batch-size 1` and `--grad-checkpoint` are the two levers when the student
does not fit, each at some cost in time.

## Read the numbers

Four numbers cover everything the reports print. A pass rate is the share of
held-out questions the checker accepted, the one figure that says whether
the task works. A loss is what training minimizes, the gap between the
student's next-token choices and the teacher's, and only its trend matters.
Nats per token is the average surprise of the student at the tokens the
teacher wrote, lower is better, and zero means the student would have
written the same thing. Bits per byte is the same surprise per byte of text,
which lets slices of different tokenization compare.

`train` prints a line every ten steps and a validation line every 200:

```
[train] it 300 loss 0.0678 dk 0.0413 alm 0.0265 ce 0.1764 floored 0 lr 1.35e-08 254 tok/s step 8854 ms load 102 ms peak 49.7 GB active 13.4 cache 8.0
[train] it 300 val 0.0744 (best 0.07517039564856337)
```

The loss should fall through the first third of the run and then flatten.
`val` is the loss on rows held out of training, and `best` is the lowest so
far. A validation loss that rises while the training loss keeps falling
means the adapter is memorizing the rows, and fewer iterations or a lower
rank fix it. `peak` is the memory high-water mark in GB and `step` the wall
time per step.

`eval` writes a Markdown report with one table per kind of measurement. The
reply table is the one the recipe reads:

| reply slice | bpb after | bpb before | nats/token after | se | rows | positions |
|---|---|---|---|---|---|---|
| heldout-ctx | 0.2225 | None | 0.6946 | 0.0056 | 173 | census high-delta |

`after` is with the adapter and `before` without it, filled in by
`--before`. The scale for nats per token is the teacher's own figure at the
same positions, which the census report lists with and without the document.
`se` is the standard error over rows, and two adapters whose figures differ
by less than two of it are the same. The text slice table adds a `decontam`
column, the share of a slice's 64-byte windows found in the training corpus,
and a slice over one percent has its `gate` marked void because it was
trained on. The task table gives accuracy before and after on the local task
files, the chat sanity table the rate of well-formed replies, and the KL
table the distance from the teacher's stored choices in nats.

## What to expect

On the worked task, with the teacher and student on one tokenizer:

| pass rate on | untouched student | after round one | after round two |
|---|---|---|---|
| held-out questions of the trained kinds | 0.022 | 0.817 | 0.882 |
| questions of kinds never trained on | 0.017 | 0.917 | 0.925 |
| combined held-out questions | 0.000 | 0.767 | 0.783 |

The untouched student with the schema pasted into its prompt scores 0.946 on
the held-out questions, so the adapter reaches most of what pasting the
document gives. At the positions the census found, the student's nats per
token fell from 4.41 to 0.70, against the teacher's 0.41 with the schema in
view. The chat sanity rate stayed at 1.00, a math task moved within its
noise, and bits per byte on a code conversation set stayed level, so an
adapter trained this way keeps the student's general behavior. Differences
of a few hundredths between rows are sampling.

A student from another tokenizer family on the same recipe reached about a
third of the same-tokenizer gap. Measure it on your own document before
relying on it.

## A ten-minute smoke run

Run the pipeline end to end on a small pair before committing hours to a
real one. The settings below match the repository's own end-to-end test: a
Qwen3 0.6B teacher at Q8_0, the same model at Q4_K_M as the student, a few
dozen paragraphs of plain text, and 40 training steps.

```sh
mkdir smoke && printf '%s\n' "The lighthouse keeper logged every ship." "Nets dried on the quay by noon." > smoke/a.txt
gmlx distill cache --teacher Qwen3-0.6B-Q8_0.gguf --corpus smoke/ --out smoke-cache/ \
    --top-k 64 --max-len 128 --rows-per-shard 16
gmlx distill align --cache smoke-cache/ --student Qwen3-0.6B-Q4_K_M.gguf --out smoke-view/
gmlx distill train --view smoke-view/ --student Qwen3-0.6B-Q4_K_M.gguf --adapter-out smoke.gguf \
    --iters 40 --batch-size 4
gmlx distill eval --student Qwen3-0.6B-Q4_K_M.gguf --adapter smoke.gguf --before \
    --slice smoke=smoke/a.txt --max-len 128 --md smoke.md --json smoke.json
```

A jsonl with a `text` field per row works in place of the directory. The run
passes when `train` reports a falling loss and `eval` writes both reports.

## Train on plain text instead

Fixed text is the simplest corpus and needs neither `gen` nor `filter`. The
teacher reads a jsonl file with a `text` field per row, a directory of text
files, or a Hugging Face dataset id, and the student learns the teacher's
choices over someone else's words. This transfers general ability and is the
right choice when the goal is a smaller model that behaves like the larger
one on ordinary text:

```sh
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --out cache/ \
    --top-k 256 --max-len 512 --max-disk-gb 20
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --cache cache/ --slice prose=heldout-prose.txt --slice code=heldout-code.txt \
    --kld-cache cache/ --md eval.md --json eval.json
```

Rows are cut into windows of at most `--max-len` teacher tokens at word
boundaries. `--frame continue` places each window in a model turn behind a
fixed instruction, so an instruct teacher sees its own template. `eval
--cache` checks each slice against the corpus so a slice the student trained
on is marked, and a same-tokenizer `--kld-cache` reports the distance from
the teacher's stored choices. Without a prompt set, `gen --corpus` builds
continuation prompts from a text corpus instead, quoting the start of each
document under an instruction to continue it.

## When something goes wrong

The filter dropped most rows. The report names the reason per row. `budget`
means the reasoning hit `--thinking-budget`, so raise it or drop
`--thinking`. `length` means the answer hit `--max-tokens`. `verify` with
your own reason words means the teacher got the task wrong with the document
in view, and a teacher that fails most of a task cannot teach it.

`align` warned about the own-group fraction. The two tokenizers split text
differently enough that part of the teacher's output has no direct student
target. Under 0.90 the run works with less signal, and under 0.70 it
refuses. Pick a student from the teacher's family, or pass `--force` and
expect a weaker result.

The census effect is small. The document changes little of what the teacher
says on these prompts. Check that `gen` ran with `--context`, that the
questions need the document, and that the second cache used `--messages-key
student_messages`.

A pass rate near zero after training has three usual causes. Serve the
student with the adapter attached and thinking on, and check that the
checker parses the student's reply format. If the untouched student with the
document pasted in also scores low, the task is beyond the student.

`train` runs out of memory. Lower `--batch-size` to 1, add
`--grad-checkpoint`, or cache with a shorter `--max-len` so the rows are
shorter.

`train` refuses the aligned cache. The cache it was built from changed, or
the tokenizer tables do not match the student. Rerun `align`.

Other failures are in [troubleshooting](troubleshooting.md).

## Advanced settings

Every default above is a setting that worked. Each flag here changes one
thing, and the full tables are under [gmlx distill](cli.md#gmlx-distill).

The cache stores, per position, the teacher's `--top-k` most likely next
tokens with their log-probabilities, the log-probability of the token that
followed, the probability mass outside the top-k, and the mass on word
boundaries. `--resume` continues after the last verified shard, and
`--validate DIR` checks an existing cache without loading a model. A MoE
teacher larger than memory streams its experts from disk. `--routes` stores
the experts a MoE teacher chose, so `eval --kld-cache` on the teacher's own
quantization measures weight noise alone. `--hidden` stores a sketch of the
teacher's final hidden state for the `train --hs` term, which is off by
default.

A frame says where the targets are in a conversation row. `chat` targets
every assistant turn, `reply` the final one, `reply-think` the final one
from its reasoning onward, and `continue` wraps plain text in a model turn.
`--per-turn` makes one row per assistant turn with the history before it.

`align` writes `view.json`, with the row index and the train and validation
split, and `tables.safetensors`, which depends only on the tokenizer pair.
On a cross-tokenizer pair it finds the byte offsets where both tokenizations
agree on a boundary. At each one it maps the teacher's top-k onto groups of
student tokens that start with the same bytes, so a digit run or a longer
merge becomes a target for a sum of student probabilities. Between
boundaries the student is trained to match the teacher's probability of the
whole chunk of bytes. The own-group fraction in `view.json` is how much of
the teacher's mass has a direct target, and the shared-boundary fraction how
much of the text is covered. A mean own-group fraction under 0.90, or a
shared-boundary fraction under 0.50, prints a warning. An own-group fraction
under 0.70 refuses unless `--force` is given.

The training loss is a sparse KL over the top-k plus a tail bucket for the
mass outside it, so the student is never asked to put all of its probability
on the top-k. `--dk` weights that term, `--alm` the chunk term on a
cross-tokenizer pair, and `--ce` a plain cross-entropy on the teacher's
tokens, which stays at 0. `--loss paper` is the top-k term with no tail
bucket, and `--loss renorm` renormalizes both sides over the top-k.
`--lora-rank` sets the adapter's width, 128 on the 9B student, and
`--lora-alpha` its scale as alpha over rank. `--lr` is the peak rate of a
warmup into cosine decay, `--seed` fixes the batch order and the adapter's
initialization, and `--resume` restarts at the exact iteration from
`--ckpt-dir`. `--view` repeats to train on several aligned caches over one
tokenizer pair.

`--iters` follows from the training row count. An epoch is the train rows of
every aligned cache divided by `--batch-size`, rounded up, and two epochs is
the recipe's setting:

```sh
python -c 'import json,math,sys; n=sum(e["split"]=="train" for v in sys.argv[1:] for e in json.load(open(v+"/view.json"))["index"]); print(n, 2*math.ceil(n/3))' view-r1/ view-r2/
```

Memory and the measurements behind these defaults are on the [distillation
internals](internals/distill.md) page.

## Limitations

- The student is a K-quant GGUF with a LoRA adapter. Full-parameter
  training and MLX checkpoints are library features without an action.
- A remote teacher can write the corpus through `gen --base-url`, and
  the trace count then needs `--tokenizer`. It cannot be cached, since
  `cache` runs the teacher's forward pass itself and needs a local GGUF.
- Recorded routes are replayed by `eval --kld-cache` on the cache's own
  teacher. `train` does not replay them, so a MoE student trained from a
  MoE teacher of the same family learns from the teacher's outputs alone.
- The hidden-state term reads the teacher's final hidden state only. No
  intermediate layer is stored, and the sketch is fixed at cache time.
- Task files for `eval` are read from disk. Nothing is downloaded.
- One cache serves any student, but an aligned cache is bound to its
  cache and its tokenizer pair, and `train` refuses one whose cache changed.
