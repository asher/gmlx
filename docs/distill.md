# Distill a teacher into a student

This guide is for anyone who wants a small GGUF model to know something a
larger one knows, without the larger one running at inference. It walks
the `gmlx distill` actions through one worked task, says what each step
costs, and explains the numbers the reports print.

You give it a document and a list of questions about it. A large GGUF,
the teacher, reads the document. A small GGUF, the student, never sees it.
You get a LoRA adapter, a small GGUF of extra weights laid over the
student, which you attach with `--adapter`. The student then answers the
questions without the document in its prompt.

On the worked task the adapter gets 0.88 of the held-out questions right,
against 0.95 with the document pasted into every prompt, and puts no
document into any request. The same steps train a student on plain text,
or on a behavior such as answering in a fixed format.

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
- [Train on plain text or a behavior instead](#train-on-plain-text-or-a-behavior-instead)
- [When something goes wrong](#when-something-goes-wrong)
- [Advanced settings](#advanced-settings)
- [Limitations](#limitations)

## Before you start

Pick a teacher and a student from the same model family when you can.
Models split text into word pieces called tokens, and two models from one
family use the same tokenizer, so every position of the teacher's output
gives the student something to match directly. The result in
[What to expect](#what-to-expect) is that same-tokenizer case. A student
from another family still works, through the mapping described under
[Advanced settings](#advanced-settings), but reaches a fraction of the
same result.

The pipeline runs the teacher on its own, once, and never loads the
teacher and the student together. `gen` and `cache` need the teacher's
memory, and `train` needs the student's weights plus its training state,
which on the worked task was the larger figure. A 27B teacher at Q8 and
a 9B student at Q6_K fit a 64 GB machine in every step, and the figures
are in [What each step costs](#what-each-step-costs). The student must
have a chat template for conversation rows and for the chat
measurements, which any instruct GGUF has.

Six actions do the work and one measures an input. Every flag of every
action is listed under [gmlx distill](cli.md#gmlx-distill) in the CLI
reference.

| Action | What it does |
|---|---|
| `gen` | Serves the teacher and writes its replies to your prompts as a corpus. |
| `filter` | Drops replies the student should not learn from, including any your own checker rejects. |
| `cache` | Runs the teacher over the corpus once and stores, per position, which next tokens it favored and by how much. |
| `align` | Reads the cache through the student's tokenizer and writes a view, the positions and values the student is trained to match. |
| `train` | Fits a LoRA adapter on the student against one or more views and writes it as a GGUF. |
| `eval` | Scores the student with and without the adapter on held-out text and tasks. |
| `census` | Measures how much a document moves the teacher, from two caches of the same replies. |

## Write the inputs

The worked task is a database schema. The teacher reads the schema and
answers questions with SQL, the student must answer the same questions
with no schema in its prompt, and a query counts as right when it runs
against the database and returns the reference rows. Any document with a
checkable task fits the same steps, such as an API reference with tests, a
style guide with a linter, or a rulebook with a judge.

Only the teacher reads the document, one text file, `schema.md` below.
The worked one is about 3400 bytes. The database, `freight.sqlite` below,
is what the checker runs queries against and is not part of the pipeline.

The prompts are jsonl files with one question per line. Each row has an
`id`, a `messages` list that ends on a user turn, and any extra fields you
want to travel with the row. The row below carries its expected answer
under `check`, which `gen` copies onto the reply row unchanged, so the
checker finds it there:

```json
{"id": "train-00000", "family": "in_transit_hull",
 "messages": [{"role": "user", "content": "For Kestrel-class ships, how many manifests have no arrival yet?\n\nAnswer with one SQLite query in a ```sql code block and nothing else."}],
 "check": {"sql": "SELECT COUNT(*) FROM manifests m JOIN haulers h ON m.hauler_id = h.hauler_id WHERE h.hull_class = 'Kestrel' AND m.arrived_at IS NULL", "ordered": false}}
```

Write three prompt sets:

- `prompts-train.jsonl` covers every kind of question the document
  answers, with several hundred rows and at least three phrasings of
  each. The worked task used 615 rows plus 264 that combine two kinds.
- `prompts-heldout.jsonl` asks the same kinds of question in words the
  training set does not use, and measures what the student learned. Two
  hundred rows make a pass rate stable to a few points.
- `prompts-untrained.jsonl` asks kinds of question the training set never
  covers, and measures whether the student learned the document or only
  the training questions.

Include training prompts that combine two kinds of question in one, such
as a count over a join. Without them the student answers each kind and
fails the combinations. Rows that describe the document in prose rather
than using it do not help and stay out.

The checker, given to `filter` as `--verify`, is any command that reads
the surviving rows as jsonl on stdin and prints one line per row, `ok` or
a reason word. The filter runs it through the shell, so a script with
arguments works. Save the skeleton below as `check-sql.py` and make it
executable:

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

Every row the checker rejects is a row that would have taught the
student a wrong answer, so the checker is not optional. The filter
records each reason word in its report and in the rejects file.

## Check that the document matters

Before spending hours on training, measure whether the document changes
what the teacher says. The check generates the teacher's replies to the
held-out prompts with the document in view, then caches those exact
replies twice, once as written and once with the document removed from
the prompt.

`--context`, the flags' word for the document, puts it into the last user
turn on the teacher's side only. Each reply row then carries two message
lists, `messages` with the document and `student_messages` without it.
The first `cache` reads `messages`, the second reads the bare list
through `--messages-key student_messages`, and `census` compares what the
teacher thought of the same replies with and without the document.

`--frame` says where in a row the targets are, the positions the student
will be trained to match, and `reply-think` puts them at the final turn
from its reasoning trace onward. `--frame-kwargs` passes the template's
thinking switch, so the teacher reads the conversation the way it wrote
it. `--max-len` is the longest row in teacher tokens. The `gen` flags are
explained under [Round one](#round-one-the-teacher-writes-the-corpus).
The replies are cached as written, including the ones whose reasoning
trace hit `--thinking-budget`, since the census wants every position the
document moved:

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

Two rows of the census report decide. The row that starts `distillable
effect` is how far the document moves the teacher's next-token choices,
averaged over every position of the replies, in nats (defined under
[Read the numbers](#read-the-numbers)). The `positions above 1.0 nats`
row is the share of positions it moves by more than one nat, which the
report calls high-delta positions. An effect under about 0.05 nats, or a
share under one percent, means the document changes little that a
student could learn, and the run is not worth its hours. The other rows
are diagnostics, and `residual across contexts` matters only with
several documents. Keep the census JSON, since the evaluation reads it
to score the adapter at those positions.

## Round one, the teacher writes the corpus

The teacher answers the training prompts with the document in view and
the checker keeps the replies that were right. The teacher is then cached
over the kept replies, the cache is aligned to the student, and the
adapter is trained:

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
    --iters 302 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 --ckpt-dir ckpt-r1/
```

`gen` serves the teacher on a local port, sends `--concurrency` requests
at a time, and stops the server when it is done. The server's own output
goes to `r1-replies.jsonl.server.log`, and the run's settings go to
`r1-replies.jsonl.gen.json`, a sidecar that `filter` and `cache` read.
A run that stops part way resumes when rerun, since prompt ids already in
the output are skipped.

`--thinking` turns the teacher's thinking on, so each reply carries its
reasoning trace under `reasoning_content` and the student learns the
trace as well as the answer. `--thinking-budget` caps the trace at 1000
tokens and `--max-tokens` gives 320 more for the answer. `--temperature`
and `--top-p` are the sampling settings, and 0.6 with 0.95 gives varied
replies that stay on task.

A teacher without a thinking mode runs without `--thinking`, and the
caches below then take `--frame reply` instead of `reply-think`. Pair a
thinking teacher with a student that has a thinking mode of its own, or
run the teacher without `--thinking`.

`filter` runs its checks in a fixed order and names the first one a row
fails, with one reason word per dropped row:

- `length`, the reply did not reach its end of turn.
- `budget`, the reasoning trace hit `--thinking-budget`.
- `empty`, the reply has fewer than `--min-tokens` words.
- `marker`, a marker of the chat template, the fixed text a model wraps
  around each turn, leaked into the reply.
- `repeat`, lines or phrases repeat.
- `ascii`, too many non-ASCII characters, only with `--max-non-ascii`.
- `tokens`, the reply is over `--max-reply-tokens`.
- `verify`, your checker said no, with its word under `detail` in the
  rejects file.

A reply whose reasoning trace hit the budget is unfinished and is
dropped as `budget`. Expect to lose about a third of the replies at a
1000-token budget, so write more prompts than the rows you need.
`--max-reply-tokens` counts the trace and the answer together. The two
budgets allow 1320 tokens, and the cap of 1180 also drops the replies
whose trace stopped just short of the budget, which are as unfinished as
the ones that hit it. The report file has the kept and dropped counts.

`cache` runs the teacher over the kept replies. `--frame reply-think`
tells it the rows are conversations and the target starts at the final
turn's reasoning trace. `--frame-kwargs` sets the same thinking switch
`gen` used, so the teacher reads the conversation the way it wrote it.
`enable_thinking` is the variable the Qwen chat template reads for its
thinking switch, the one `gmlx serve --thinking on` sets. Another
family's template may name it differently, and `gmlx serve --help` under
`--thinking` says which.

`--max-len` is the longest row in teacher tokens, and 2560 holds a
1180-token reply behind a prompt. `--top-k` is how many next-token
candidates are stored per position, 256 by default, and is unrelated to
the sampler's `--top-k` on `gen`.

`align` runs on the CPU with the two tokenizers only and writes the view
into `view-r1/`. Its log line ends with `a` and `s`, the own-group and
shared-boundary fractions explained under
[When something goes wrong](#when-something-goes-wrong). On the worked
pair every teacher token has a student token of its own, `a=1.000`, and
577 rows took about 15 seconds.

`train` reads the view, fits the adapter on the quantized student, and
writes the adapter GGUF. `--iters` counts steps of `--batch-size` rows
each, and two passes over the rows is enough for a generated corpus.
Count the rows once `align` has written the view, with the batch size of
3 written in:

```sh
python -c 'import json,math,sys; n=sum(e["split"]=="train" for v in sys.argv[1:] for e in json.load(open(v+"/view.json"))["index"]); print(n, 2*math.ceil(n/3))' view-r1/
```

302 is that figure for a view of 453 training rows. `--lora-rank` is
the adapter's capacity, `--lora-alpha` its scale, and `--lr` the peak
learning rate, how far each step moves the adapter. `--ckpt-dir` is
where the run keeps its checkpoints, `./ckpt` by default. The values
above are the ones that worked on the 9B student, and
[Advanced settings](#advanced-settings) says what each one changes.

If the loss printed every ten steps has not fallen by the fortieth step,
stop the run. Check that the filter kept the rows you expected and that
`align` reported `a=1.000` or a warning you accepted.

## Use and measure the adapter

The adapter attaches to the same student GGUF it was trained on. It is
tied to that file, and a different quantization of the same model is a
different student:

```sh
gmlx serve Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --thinking on
```

A student trained with `--frame reply-think` learned to reason before
answering, so serve it with thinking on. The Qwen template turns it on by
default, and `--thinking on` makes sure. The switch holds for every
request, and a request turns it off with
`"chat_template_kwargs": {"enable_thinking": false}` in its body, listed
under [parameter support](api.md#parameter-support).

The server registers the adapted model under an id derived from the file
name, `qwen3.5-9b` here, and the bare base as `<id>-base`. Both are
listed by `curl localhost:8080/v1/models`, and a request names one of
them:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen3.5-9b",
  "messages": [{"role": "user", "content": "How many manifests have no arrival yet?"}]
}'
```

[Use the adapter](lora.md#use-the-adapter) in the LoRA guide covers `run`
and `chat`, and its [interop section](lora.md#adapter-format-and-interop)
says how the same file loads in llama.cpp.

The pass rate uses the same checker that filtered the corpus. Generate
the student's replies to the held-out prompts, without the document, and
run the checker over them:

```sh
gmlx distill gen --teacher Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-heldout.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-r1.jsonl
gmlx distill filter --in heldout-r1.jsonl --out heldout-r1-ok.jsonl \
    --verify "./check-sql.py freight.sqlite" --report heldout-r1.json
```

The `--teacher` flag names whatever model `gen` serves, here the student
with its adapter. `--serve-arg` passes an argument through to
`gmlx serve`, and two of them attach the adapter. The `=` form is needed
because the value starts with `--`. Requests go to the adapted model,
which the server lists first.

The pass rate is `kept` divided by `kept` plus the sum of the `dropped`
counts in the report. Run the same two commands on the untouched
student, without the serve arguments, for the before figure, and on
`prompts-untrained.jsonl` for the third pass rate.

`eval` scores what a pass rate cannot see. With the census JSON from the
check above it scores the adapter at the positions the document moved,
and it checks that the student's general behavior survived:

```sh
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --before \
    --reply-slice heldout=heldout-ctx.jsonl --reply-think --reply-positions census.json \
    --md eval.md --json eval.json
```

`--before` scores the same loaded model a second time with the adapter
switched off, so both figures come from one process. `--reply-slice`
scores the teacher's held-out replies from the census check on the bare
prompt under `student_messages`, so the document is not in view.
`--reply-think` includes their reasoning trace, and `--reply-positions`
keeps only the positions the document moved.
[Read the numbers](#read-the-numbers) explains the report.

## Round two, the student writes and the teacher scores

A second round trains the student on the teacher's corrections of its own
replies, and lifts the pass rate a few points over round one. The
student with its first adapter answers the training prompts without the
document and the checker keeps the right replies. The filter then puts
the document back on the teacher's side before the teacher is cached
over them:

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
    --adapter-out r2.gguf --iters 678 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 \
    --ckpt-dir ckpt-r2/
```

`filter --context` rebuilds every kept row with the document in the
teacher's prompt and the bare prompt under `student_messages`. The cache
then holds what the teacher thinks of the student's own words with the
document in view, while the student trains on the prompt alone.
`align --tables` reuses the tokenizer-pair tables, `tables.safetensors`,
from round one.

`train` takes both views and starts from the base weights, not from the
first adapter, so `--iters` doubles with the rows. The one-liner under
Round one counts them when given both view directories. `--ckpt-dir`
keeps the two rounds' checkpoints apart, since both would write to
`./ckpt` without it. Measure `r2.gguf` the same way as round one.

## What each step costs

The figures below are from the worked task, with a 27B teacher at Q8, a
9B student at Q6_K, 615 training questions plus 264 combined ones, and
two rounds. The whole run took about 11 hours.

| Step | Memory | Time | Disk |
|---|---|---|---|
| `gen` | the teacher, served | limited by generation speed, so `--concurrency` 8 keeps it busy | the replies, a few MB |
| `cache` | the teacher plus a few GB | about 500 teacher tokens per second | 1.6 KB per position at top-k 256 |
| `align` | the tokenizers only | about 25 ms per conversation row on the CPU, under 1 ms per plain-text row | a few MB unless `--materialize` |
| `train` | 50.7 GB peak on the 9B student | 9.3 s per step of 3 rows, so 678 steps in 1.75 h | two checkpoints under `--ckpt-dir`, each the adapter's parameters plus two optimizer moments of the same size |
| `eval` | the student | minutes per slice, longer with the adapter attached | two report files |

A cache is 6 x K + 22 bytes per position plus the text, with K the
`--top-k` value, so a corpus of 600 rows near 1300 tokens takes about
1.2 GB at the default K.
`--max-disk-gb` refuses a cache whose estimate exceeds it, and every size
flag counts decimal GB. When a run is over, the caches, the checkpoint
directories, the server logs and the sidecars can all go. The views, the
adapter and the reports are what you keep.

Training memory scales with the row length and the batch. `--batch-size
1` and `--grad-checkpoint` are the two levers when the student does not
fit, each at some cost in time.

A teacher that does not fit is a different problem. Pick a smaller
quantization of it, or stream a MoE teacher's experts from disk, which
`cache` does on its own for a MoE model larger than memory. A MoE model,
a mixture-of-experts model, has layers split into experts of which a
few run per token, and `gmlx validate` says whether a file is one. `gen`
alone can also use a teacher served elsewhere through `--base-url`. A
teacher that fits nowhere local cannot be cached, as
[Limitations](#limitations) says.

The document's length is bounded by the teacher's context window during
`gen` and by `--max-len` during `cache`, since the document sits in the
teacher's prompt on every row. A few thousand tokens, at roughly four
bytes of English text per token, is comfortable at the settings above.
A longer document raises `--max-len` and the memory of every teacher
step with it.

## Read the numbers

Four numbers cover everything the reports print:

- A pass rate is the share of held-out questions the checker accepted,
  the one figure that says whether the task works. One served run moves
  it by three or four items in a hundred, so differences of that size
  between adapters mean nothing.
- A loss is what training minimizes, the gap between the student's
  next-token choices and the teacher's. Only its trend matters.
- Nats per token is the average surprise of the student at the tokens
  the teacher wrote. A nat is the natural log of one over the
  probability the student gave the token, so a token given probability
  0.37 costs one nat. Lower is better, and zero means the student would
  have written the same thing.
- Bits per byte, `bpb` in the tables, is the same surprise per byte of
  text, which lets slices of different tokenization compare.

`train` prints a line every ten steps and a validation line every 200
steps and at the last step:

```
[train] it 200 loss 0.0637 dk 0.0420 alm 0.0217 ce 0.1584 floored 0 lr 4.23e-05 249 tok/s step 9858 ms load 111 ms peak 50.7 GB active 13.4 cache 8.0
[train] it 200 val 0.0795 (best 0.0900)
```

The fields of the train line:

- `loss` should fall through the first third of the run and then flatten.
- `dk` and `alm` are its two parts. `alm` prints 0 on a view whose
  tokenizers `align` recorded as matching, where that term is off.
- `ce` is a third term that is measured but not trained on unless `--ce`
  is set.
- `floored` counts positions whose probability was clamped at the
  smallest representable value.
- `lr` is the learning rate at that step.
- `tok/s`, `step` and `load` are the throughput, the wall time per step
  and the time spent reading the batch.
- `peak`, `active` and `cache` are memory in GB, the high-water mark, the
  arrays in use and MLX's buffer cache.

`val` is the loss on rows held out of training, and `best` is the lowest
of the earlier validations, so a `val` below it is a new best. A
validation loss that rises while the training loss keeps falling means
the adapter is memorizing the rows, and fewer steps or a lower rank fix
it.

`eval` writes a Markdown report with one table per kind of measurement,
each row a slice, one held-out file. The reply table is the one this
guide reads:

| reply slice | bpb after | bpb before | nats/token after | se | rows |
|---|---|---|---|---|---|
| heldout | 0.2225 | 1.4131 | 0.6946 | 0.0056 | 173 |

`after` is with the adapter and `before` without it, filled in by
`--before`. With `--reply-positions` the rows are scored at the
positions the document moved only, and the log line says so. Judge the
nats per token figure against the teacher's own at the same positions,
in the census report's `teacher nats per token on high-delta positions`
row. `se` is the standard error over rows, and two adapters whose
figures differ by less than two of it cannot be told apart.

The other tables follow the same after and before pattern. The slice
table adds a `decontam` column, the share of a slice's 64-byte windows
found in the training corpus, and a slice over one percent has its
`gate`, its standing as a pass mark, set to void because it was trained
on. The task table gives accuracy on the local task files. The chat
sanity table gives the share of replies that kept the turn structure as
`compliance`, the share the budget cut as `truncated_rate`, and the
share that read as refusals. The KL table gives the KL divergence, a
distance between the student's next-token distribution and the
teacher's stored one, in nats.

## What to expect

On the worked task, with the teacher and student on one tokenizer:

| pass rate on | untouched student | after round one | after round two |
|---|---|---|---|
| held-out questions of the trained kinds | 0.022 | 0.817 | 0.882 |
| questions of kinds never trained on | 0.017 | 0.917 | 0.925 |
| combined held-out questions | 0.000 | 0.767 | 0.783 |

The untouched student with the schema pasted into its prompt scores
0.946 on the held-out questions, so the adapter reaches most of what
pasting the document gives. At the positions the document moved, the
student's nats per token fell from 4.41 to 0.70, against the teacher's
0.41 with the schema in view.

Whether the student's general behavior survived is measured by the
`eval` flags that score a chat prompt set, a conversation set and local
task files, `--chat-sanity`, `--chat-slice` and `--tasks` under
[distill eval](cli.md#distill-eval). On the worked task none of those
moved outside its noise.

A student from another tokenizer family on the same task reached about
a third of the same-tokenizer gap. Measure it on your own document
before relying on it.

## A ten-minute smoke run

Run the pipeline end to end on a small pair before committing hours to a
real one. The settings below use a Qwen3 0.6B teacher at Q8_0, the same
model at Q4_K_M as the student, a jsonl of a few dozen paragraphs with
one `{"text": ...}` row per paragraph, and 40 training steps. Any text
works, but `train` needs more rows than `--batch-size` and `align` keeps
one row back for validation, so give it twenty or more:

```sh
python -c 'import json; [print(json.dumps({"text": f"Paragraph {i}. The lighthouse keeper logged every ship that passed the point."})) for i in range(24)]' > smoke.jsonl
python -c 'import json,sys; [print(json.loads(l)["text"]) for l in open("smoke.jsonl")]' > smoke.txt
gmlx distill cache --teacher Qwen3-0.6B-Q8_0.gguf --corpus smoke.jsonl --out smoke-cache/ \
    --top-k 64 --max-len 128 --rows-per-shard 16
gmlx distill align --cache smoke-cache/ --student Qwen3-0.6B-Q4_K_M.gguf --out smoke-view/
gmlx distill train --view smoke-view/ --student Qwen3-0.6B-Q4_K_M.gguf --adapter-out smoke.gguf \
    --iters 40 --batch-size 4
gmlx distill eval --student Qwen3-0.6B-Q4_K_M.gguf --adapter smoke.gguf --before \
    --slice smoke=smoke.txt --max-len 128 --md smoke.md --json smoke.json
```

A directory of text files works in place of the jsonl, one row per file,
and `--slice` takes plain text. The slice is the training text on
purpose, so the after figure must beat the before one. The run passes
when `train` reports a falling loss and `eval` writes both reports.

## Train on plain text or a behavior instead

Fixed text is the simplest corpus and needs neither `gen` nor `filter`.
The teacher reads a jsonl file with a `text` field per row, a directory
of text files, or a Hugging Face dataset id, and the student learns the
teacher's choices over someone else's words. This transfers general
ability and is the right choice when the goal is a smaller model that
behaves like the larger one on ordinary text:

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
boundaries, and without a `--frame` they are cached as plain text. An
instruct teacher sees its own template with `--frame continue`, which
places each window in an assistant turn behind a fixed
`--frame-instruction`. `--iters 2000` at the default batch of 8 is two
passes over 8000 rows of 512 tokens. Size the corpus by the validation
line. A `val` that stops falling while `loss` keeps falling wants more
text, and a `val` still falling at the last step wants more steps.

`eval --cache` checks each slice against the corpus so a slice the
student trained on is marked, and a same-tokenizer `--kld-cache` reports
the distance from the teacher's stored choices.

Without a prompt set, `gen --corpus` builds continuation prompts from a
text corpus instead, quoting the start of each document under an
instruction to continue it.

A behavior, such as answering in a fixed format, is round one without a
document. Write prompts that call for the behavior, run `gen` without
`--context`, and give `filter` a checker for the format, or no `--verify`
at all when the teacher's replies are the standard. The rest of the
round is unchanged.

## When something goes wrong

The filter dropped most rows. The rejects file names the reason per row.
`budget` means the reasoning trace hit `--thinking-budget`, so raise it or drop
`--thinking`. `length` means the answer hit `--max-tokens`. `verify` with
your own reason words means the teacher got the task wrong with the
document in view, and a teacher that fails most of a task cannot teach
it.

`align` printed `warn: own-group fraction a=...`. The two tokenizers
split text differently enough that part of the teacher's output has no
direct student target. Under 0.90 the run works with less signal, and
under 0.70 it refuses. Pick a student from the teacher's family, or pass
`--force` and expect a weaker result.

The census effect is small. The document changes little of what the
teacher says on these prompts. Check that `gen` ran with `--context`,
that the questions need the document, and that the second cache used
`--messages-key student_messages`.

A pass rate near zero after training has three usual causes. The student
was served without the adapter or with thinking off, so serve it with
`--adapter` and thinking on. The checker cannot parse the student's reply
format, so check a few replies by hand. The task is beyond the student,
which shows when the untouched student with the document pasted in also
scores low, and no adapter will fix that.

`train` runs out of memory. Lower `--batch-size` to 1, add
`--grad-checkpoint`, or cache with a shorter `--max-len` so the rows are
shorter.

A refused view means the cache it was built from changed, or the
tokenizer tables do not match the student, so rerun `align`. A refusal
naming fewer train rows than `--batch-size` means the corpus is too
small for that batch.

Other failures are in [troubleshooting](troubleshooting.md).

## Advanced settings

Each flag here changes one thing, and the full tables are under
[gmlx distill](cli.md#gmlx-distill).

The cache stores, per position, the teacher's `--top-k` most likely next
tokens with their log-probabilities and the log-probability of the token
that followed. It also stores the probability mass outside the top-k,
the probability summed over every other token, and the mass on word
boundaries. `--resume` continues after the last verified shard, and
`--validate DIR` checks an existing cache without loading a model.

Two cache flags serve MoE teachers. A MoE teacher larger than memory
streams its experts from disk. `--routes` stores the experts the teacher
chose, so `eval --kld-cache` on the teacher's own quantization measures
weight noise alone. `--hidden` stores a sketch of the teacher's final
hidden state, a fixed-width compressed copy of it, for the `train --hs`
term, which is off by default.

A frame says where the targets are in a conversation row. `chat`
targets every assistant turn, `reply` the final one, `reply-think` the
final one from its reasoning trace onward, and `continue` wraps plain
text in an assistant turn. `--per-turn` makes one row per assistant
turn with the history before it. `--context-format` on `gen` and
`filter` decides how the document and the question combine in the
teacher's prompt, and a row's own `context` field takes precedence over
`--context`, which is how several documents share one run.

`align` writes `view.json`, with the row index and the train and
validation split, and `tables.safetensors`, which depends only on the
tokenizer pair. A view is bound to the cache it was built from and to
that pair.

On a cross-tokenizer pair `align` finds the byte offsets where both
tokenizations agree on a boundary. At each one it maps the teacher's
top-k onto groups of student tokens that start with the same bytes, so a
digit run or a longer merge becomes a target for a sum of student
probabilities. Between boundaries the student is trained to match the
teacher's probability of the whole chunk of bytes. The own-group
fraction in `view.json` is how much of the teacher's mass has a direct
target, and the shared-boundary fraction how much of the text is
covered. A mean own-group fraction under 0.90, or a shared-boundary
fraction under 0.50, prints a warning. An own-group fraction under 0.70
refuses unless `--force` is given.

The training loss is a sparse KL over the top-k plus a tail bucket for
the mass outside it, so the student is never asked to put all of its
probability on the top-k. `--dk` weights that term, printed as `dk` in
the train line. `--alm` weights the chunk term of a cross-tokenizer pair,
ALM in the flags, printed as `alm`. `--ce` weights a plain cross-entropy
on the teacher's tokens, printed as `ce` and left at 0. `--loss paper`
is the top-k term with no tail bucket, and `--loss renorm` renormalizes
both sides over the top-k.

`--lora-rank` sets the adapter's width, 128 on the 9B student, and
`--lora-alpha` its scale as alpha over rank. `--lr` is the peak learning
rate. The rate rises over the first `--warmup` fraction of the steps and
then falls along a cosine curve to zero. Those are the only measured
values, so on a student of another size start from them and change one
at a time, judged by the validation loss. `--seed` fixes the batch order
and the adapter's initialization, and `--resume` restarts at the exact
step from `--ckpt-dir`. `--view` repeats to train on several views over
one tokenizer pair.

`--iters` follows from the training row count. One pass over the rows,
an epoch, takes the train rows of every view divided by `--batch-size`
steps, rounded up, and the setting above is two passes. The one-liner
under [Round one](#round-one-the-teacher-writes-the-corpus) counts them.

Memory and the measurements behind these defaults are on the
[distillation internals](internals/distill.md) page.

## Limitations

- The student is a K-quant GGUF, a file in one of the K-quant formats
  such as Q4_K_M or Q6_K, with a LoRA adapter. [Why train on the
  quant](lora.md#why-train-on-the-quant) says why. Full-parameter
  training and MLX checkpoints are library features without an action.
- An adapter is trained from the base weights each time. A changed
  document means a new round one, not a top-up of the old adapter.
- A remote teacher can write the corpus through `gen --base-url`, and
  counting its reasoning against `--thinking-budget` then needs
  `--tokenizer`. It cannot be cached, since
  `cache` runs the teacher's forward pass itself and needs a local GGUF.
- Recorded routes are replayed by `eval --kld-cache` on the cache's own
  teacher. `train` does not replay them, so a MoE student trained from a
  MoE teacher of the same family learns from the teacher's outputs alone.
- The hidden-state term reads the teacher's final hidden state only. No
  intermediate layer is stored, and the sketch is fixed at cache time.
- Task files for `eval` are read from disk. Nothing is downloaded.
- One cache serves any student, but a view is bound to its cache and its
  tokenizer pair, and `train` refuses one whose cache changed.
