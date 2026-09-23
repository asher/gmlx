# Distill a teacher into a student

This guide is for anyone who wants a small model to know something a
larger one knows, without running the larger one when the small one
answers. Both are GGUF files, the model format gmlx serves. The guide
walks the `gmlx distill` actions through one worked task, says what
each step costs, and explains the numbers the reports print.

You give it a document and a list of questions about it. A large GGUF,
the teacher, reads the document. A small GGUF, the student, never sees it.
You get a LoRA adapter, a small GGUF of extra weights laid over the
student, which you attach with `--adapter`. The student then answers the
questions without the document in its prompt.

On the worked task the student with the adapter answers 0.88 of the
held-out questions right, questions it never saw in training. The same
student with the document pasted into every prompt answers 0.95, and
with neither it answers almost none. With the adapter, no request
carries the document.

The same steps train a student on plain text, or on a behavior such as
answering in a fixed format.

- [Before you start](#before-you-start)
- [A ten-minute smoke run](#a-ten-minute-smoke-run)
- [Write the inputs](#write-the-inputs)
- [Check that the document matters](#check-that-the-document-matters)
- [Round one trains on the teacher's replies](#round-one-trains-on-the-teachers-replies)
- [Use and measure the adapter](#use-and-measure-the-adapter)
- [Round two trains on the student's own right answers](#round-two-trains-on-the-students-own-right-answers)
- [What each step costs](#what-each-step-costs)
- [Read the numbers](#read-the-numbers)
- [What to expect](#what-to-expect)
- [Train on plain text or a behavior instead](#train-on-plain-text-or-a-behavior-instead)
- [When something goes wrong](#when-something-goes-wrong)
- [Advanced settings](#advanced-settings)
- [Limitations](#limitations)

## Before you start

Models split text into word pieces called tokens, and the rule that
does the splitting is the tokenizer. Two models from one family, such as
Qwen3.5 and Qwen3.6, share a tokenizer, so the student can match the
teacher token for token. A position, in the reports, is one token of
the text.

Pick a teacher and a student from the same family when you can. The
result in [What to expect](#what-to-expect) is that same-tokenizer case.
A student from another family still works, through the mapping described
under [Advanced settings](#advanced-settings), but reaches a fraction of
the same result.

Check a pair in minutes before hours of caching. Run `cache --max-rows 8`
on any corpus, then `align` with the student, and read `a` on the
`align` summary line, where 1.000 is the ideal. The smoke run below
shows both commands.

Both models are GGUF files on disk. `gmlx pull` downloads one from a
Hugging Face repository into your model directory, as
[gmlx pull](cli.md#gmlx-pull) describes, and `--to .` saves it in the
current directory instead, which is where the commands below expect it.
The worked task pulls its two files this way:

```sh
gmlx pull hf:unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf --to .
gmlx pull hf:unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q6_K.gguf --to .
```

Every `--teacher` and `--student` below is a file path, so a file kept
elsewhere is named by its full path. A file name carries the model's
size in parameters, 27B or 9B, and its quantization, Q8 or Q6_K, the
precision its weights were shrunk to. A `UD-` prefix marks Unsloth's
mixed recipe, which keeps some tensors at higher precision. Without
`--to`, `pull` also registers the file in your gmlx config unless
`--no-register` is given.

The pipeline runs the teacher on its own, once, and never loads the
teacher and the student together. `gen` and `cache` need the teacher's
memory, and `train` needs the student's weights plus its training state.
On the worked task `train` needed more memory than `cache`, about 51 GB
against 40 GB with a 27B teacher at Q8 and a 9B student at Q6_K. A Mac
with 64 GB has that with nothing else large running, and the figures are
in [What each step costs](#what-each-step-costs). The smoke run in the
next section fits any Apple Silicon Mac.

Conversation rows and the chat measurements need a student with a chat
template, the fixed text a model wraps around each turn of a
conversation. Chat models carry one. Gemma adds `-it` to their names,
many families add `Instruct`, and Qwen3.5 and Qwen3.6 chat models, like
the worked student, carry no suffix.

Six actions do the work and one checks whether the document is worth
training on. Every flag of every action is listed under
[gmlx distill](cli.md#gmlx-distill) in the CLI reference.

| Action | What it does |
|---|---|
| `gen` | Serves the teacher and writes its replies to your prompts as a corpus, the file of rows the student trains on. |
| `filter` | Drops replies the student should not learn from, including any your own checker rejects. |
| `cache` | Runs the teacher over the corpus once and stores, per position, which next tokens it favored and by how much. |
| `align` | Reads the cache through the student's tokenizer and writes a view, the positions and values the student is trained to match. |
| `train` | Fits a LoRA adapter on the student against one or more views and writes it as a GGUF. |
| `eval` | Scores the student with and without the adapter on held-out text and tasks. |
| `census` | Checks how much a document moves the teacher, from two caches of the same replies. |

## A ten-minute smoke run

Run the pipeline end to end on a small pair before committing hours to a
real one. The run exercises `cache`, `align`, `train` and `eval` on a
Qwen3 0.6B teacher at Q8_0 and the same model at Q4_K_M as the student,
and trains for 80 steps.

Its corpus is a jsonl file, one JSON object per line, of 24
`{"text": ...}` rows. A step trains on one batch of `--batch-size` rows,
so `train` needs at least that many training rows. `align` also holds
back about one row in fifty for validation, whole documents at a time and
at least one row, and never trains on them. A cache made from one
document splits that document. Here 24 rows leave 23 for a batch of 4.
`eval --slice` reads plain text, so the second line writes the same rows
to `smoke.txt`:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --to .
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
python3 -c 'import json; [print(json.dumps({"text": f"Paragraph {i}. The lighthouse keeper logged every ship that passed the point."})) for i in range(24)]' > smoke.jsonl
python3 -c 'import json,sys; [print(json.loads(l)["text"]) for l in open("smoke.jsonl")]' > smoke.txt
gmlx distill cache --teacher Qwen3-0.6B-Q8_0.gguf --corpus smoke.jsonl --out smoke-cache/ \
    --top-k 64 --max-len 128
gmlx distill align --cache smoke-cache/ --student Qwen3-0.6B-Q4_K_M.gguf --out smoke-view/
gmlx distill train --view smoke-view/ --student Qwen3-0.6B-Q4_K_M.gguf --adapter-out smoke.gguf \
    --iters 80 --batch-size 4
gmlx distill eval --student Qwen3-0.6B-Q4_K_M.gguf --adapter smoke.gguf --before \
    --slice smoke=smoke.txt --max-len 128 --md smoke.md --json smoke.json
```

A directory of text files also works as the corpus, one row per file. A
slice is a held-out text file that `eval` scores. Here it is the
training text on purpose, so `bpb after` in `smoke.md`, the student's
bits per byte with the adapter, must come out below `bpb before`. The
run passes when the `[train] it` lines show the loss, the figure
training drives down, falling, and `eval` writes both reports.
`--top-k` and `--max-len` on `cache` shrink the cache for a quick run,
`--max-len` on `eval` scores the slice in 128-token windows, and
`--before` also scores the student with the adapter off.

The served path is checked the same way, since `gen` and `filter` run on
the same pair in under a minute:

```sh
python3 -c 'import json; [print(json.dumps({"id": f"smoke-{i}", "messages": [{"role": "user", "content": f"Describe lighthouse number {i} in one sentence."}]})) for i in range(2)]' > smoke-prompts.jsonl
gmlx distill gen --model Qwen3-0.6B-Q8_0.gguf --prompts smoke-prompts.jsonl --max-tokens 96 \
    --out smoke-replies.jsonl
gmlx distill filter --in smoke-replies.jsonl --out smoke-corpus.jsonl --min-words 1
```

Both pass when `gen` ends with a `[gen] done:` line that reports
`0 failed` and `filter` prints `[filter] kept 2`. `gen` serves the
teacher with its thinking off unless `--thinking` is given, so a
96-token reply reaches its end. `--model` names the
GGUF `gen` serves and is the same flag as `--teacher`, and
`--min-words 1` keeps one-sentence replies that the default of 16 words
would drop.

## Write the inputs

You bring three things, a document, questions about it, and a way to
check an answer. The worked files below are not shipped, so read their
names as placeholders for your own.

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
want to travel with the row, such as `family` below. Ids must be unique,
since a rerun of `gen` skips the ids already in its output. The row
below carries its expected answer under `check`, which `gen` copies onto
the reply row unchanged, so the checker finds it there:

```json
{"id": "train-00000", "family": "in_transit_hull",
 "messages": [{"role": "user", "content": "For Kestrel-class ships, how many manifests have no arrival yet?\n\nAnswer with one SQLite query in a ```sql code block and nothing else."}],
 "check": {"sql": "SELECT COUNT(*) FROM manifests m JOIN haulers h ON m.hauler_id = h.hauler_id WHERE h.hull_class = 'Kestrel' AND m.arrived_at IS NULL", "ordered": false}}
```

Write four prompt sets, one to train on and three to measure with:

- `prompts-train.jsonl` covers every kind of question the document
  answers, with several hundred rows and at least three phrasings of
  each. The worked task used 615 rows plus 264 that combine two kinds.
- `prompts-heldout.jsonl` asks the same kinds of question in words the
  training set does not use, and measures what the student learned. Two
  hundred rows make a pass rate stable to a few points.
- `prompts-untrained.jsonl` asks kinds of question the training set never
  covers, and measures whether the student learned the document or only
  the training questions.
- `prompts-heldout-combined.jsonl` holds held-out questions that combine
  two kinds, scored on their own because they are the hardest.

Every set carries the `check` field, since the pass rate runs the
checker over the replies to the three measurement sets too.

Include training prompts that combine two kinds of question in one, such
as a count over a join. Without them the student answers each kind and
fails the combinations. Rows that describe the document in prose rather
than using it do not help and stay out.

Hundreds of prompts are easiest to write as templates, one question
shape per kind, filled from the database's own values by a script that
also writes the reference query under `check`. The skeleton below has
one shape with three phrasings. Save it as `make-prompts.py` next to
`freight.sqlite` and run it with `python3`, redirecting its output to
`prompts-train.jsonl`. Add a shape per kind of question, and shapes that
combine two kinds. For the held-out file, run it again with fresh
phrasings and the id prefix changed to `heldout-`, and write the combined
shapes to their own file. A combined shape joins two kinds in one
question and one reference query, such as "How many {hull}-class
manifests left port in {year} and have not arrived?" with a query that
applies both conditions. For `prompts-untrained.jsonl`, write shapes for
kinds the training file leaves out entirely:

```python
#!/usr/bin/env python3
import json, sqlite3
db = sqlite3.connect("file:freight.sqlite?mode=ro", uri=True)
suffix = "\n\nAnswer with one SQLite query in a ```sql code block and nothing else."
shapes = {"in_transit_hull": (
    ["For {hull}-class ships, how many manifests have no arrival yet?",
     "How many {hull}-class manifests are still in transit?",
     "Count the manifests without an arrival for {hull} hulls."],
    "SELECT COUNT(*) FROM manifests m JOIN haulers h ON m.hauler_id = h.hauler_id "
    "WHERE h.hull_class = '{hull}' AND m.arrived_at IS NULL")}
hulls = [r[0] for r in db.execute("SELECT DISTINCT hull_class FROM haulers")]
n = 0
for family, (phrasings, sql) in shapes.items():
    for hull in hulls:
        for text in phrasings:
            print(json.dumps({"id": f"train-{n:05d}", "family": family,
                              "messages": [{"role": "user", "content": text.format(hull=hull) + suffix}],
                              "check": {"sql": sql.format(hull=hull), "ordered": False}}))
            n += 1
```

Given to `filter` as `--verify`, the checker is any command that reads
the surviving rows as jsonl on stdin and prints one line per row, `ok` or
a reason word. The filter runs it through the shell, so a script with
arguments works. Save the skeleton below as `check-sql.py` and run
`chmod +x check-sql.py`. It opens the database read-only, so a reply
that deletes rows cannot damage the reference data:

```python
#!/usr/bin/env python3
import json, sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for line in sys.stdin:
    row = json.loads(line)
    reply = row["messages"][-1]["content"]
    sql = reply.split("```sql", 1)[-1].split("```", 1)[0].strip()
    try:
        got = db.execute(sql).fetchall()
        want = db.execute(row["check"]["sql"]).fetchall()
        if not row["check"].get("ordered"):
            got, want = sorted(map(repr, got)), sorted(map(repr, want))
        print("ok" if got == want else "wrong_rows")
    except Exception:
        print("bad_sql")
```

Every row the checker rejects is a row that would have taught the
student a wrong answer, so the checker is not optional. The report
counts dropped rows per reason, with every checker rejection under
`verify`, and the rejects file gives the checker's own word under
`detail`.

A task with no mechanical check can use a judge in the same place, a
script that asks a served model whether the reply matches a reference
answer you wrote under `check.answer`. The one below asks whatever
`gmlx serve` has on port 8080. Any instruct model larger than the
student serves as the judge, the teacher included. Save it as
`judge.py`, run
`chmod +x judge.py`, and give it to `filter` as `--verify ./judge.py`.
`filter` runs after `gen` has stopped its own server, so the judge has
the memory to itself. Start it with `gmlx serve <judge>.gguf` before
`filter` and stop it with `gmlx stop` after:

```python
#!/usr/bin/env python3
import json, sys, urllib.request
for line in sys.stdin:
    row = json.loads(line)
    q = ("Reference answer:\n" + row["check"]["answer"] + "\n\nCandidate answer:\n"
         + row["messages"][-1]["content"] + "\n\nDoes the candidate say the same? Reply ok or wrong.")
    body = json.dumps({"messages": [{"role": "user", "content": q}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    print("ok" if reply.strip().lower().startswith("ok") else "wrong")
```

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

`--thinking` turns the teacher's thinking on and `--thinking-budget`
caps the reasoning trace, the text a thinking model writes before its
answer. [Round one](#round-one-trains-on-the-teachers-replies) explains
the other flags. The replies are cached as written, including the ones
whose trace hit the budget, since the census wants every position the
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

Two rows of the census report decide. The row that starts
`distillable effect` is how far the document moves the teacher's
next-token choices, averaged over every position of the replies, in
nats (defined under [Read the numbers](#read-the-numbers)). The
`high-delta positions` row is the share of positions where the document
makes the token the teacher wrote more likely by more than one nat.

An effect under about 0.05 nats, or a high-delta share under 0.01,
means the document changes little that a student could learn, and the
run is not worth its hours. The other rows are diagnostics. The
`paired reply rows` row also counts reply mismatches skipped, rows whose
reply bytes differed between the two caches. With several `--with`
caches, every cache decides which rows pair and which positions count,
the effect, the histogram and the positions map come from the first,
and `residual across contexts` alone reads them all. Keep the census
JSON, which the evaluation reads to score the adapter at those
positions.

The same replies give the teacher's own pass rate, computed as under
[Use and measure the adapter](#use-and-measure-the-adapter). A teacher
that fails most questions with the document in view cannot teach them.
These two commands compute it:

```sh
gmlx distill filter --in heldout-ctx.jsonl --out heldout-ctx-ok.jsonl \
    --min-words 1 --verify "./check-sql.py freight.sqlite" --report heldout-ctx-filter.json
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["kept"]/(r["kept"]+sum(r["dropped"].values())))' heldout-ctx-filter.json
```

## Round one trains on the teacher's replies

The teacher answers the training prompts with the document in view and
the checker keeps the replies that were right. The teacher is then cached
over the kept replies, the cache is aligned to the student, and the
adapter is trained. Two values below depend on your data,
`--max-reply-tokens` on `filter` and `--iters` on `train`, and the
one-liners later in this section size them. Run the commands one at a
time and size each value before the command that uses it:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-train.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r1-replies.jsonl
gmlx distill filter --in r1-replies.jsonl --out r1-corpus.jsonl \
    --min-words 1 --max-reply-tokens 1180 --verify "./check-sql.py freight.sqlite" \
    --report r1-filter.json --rejects r1-rejects.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r1-corpus.jsonl --out cache-r1/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r1/ --student Qwen3.5-9B-Q6_K.gguf --out view-r1/
gmlx distill train --view view-r1/ --student Qwen3.5-9B-Q6_K.gguf --adapter-out r1.gguf \
    --iters 300 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 --ckpt-dir ckpt-r1/
```

`gen` serves the teacher on port 8093, its own default so that a run
never collides with a `gmlx serve` on 8080, sends `--concurrency`
requests at a time, and stops the server when it is done. The server's
own output goes to `r1-replies.jsonl.server.log`, and the run's settings
go to
`r1-replies.jsonl.gen.json`, a sidecar that `filter` and `cache` read.

Each action resumes in its own way. A `gen` that stops part way resumes
when rerun, since prompt ids already in the output are skipped, and it
refuses when a skipped id no longer names the prompt it answered.
`cache --resume` continues after the last shard, a file of 64 cached
rows, that it wrote and verified, and refuses when the corpus or the row
flags differ from the first run. `train --resume` continues from the
last checkpoint under `--ckpt-dir` and refuses when there is none or
when the run's settings changed since the checkpoint was written.

`--thinking` turns the teacher's thinking on, so each reply carries its
reasoning trace under `reasoning_content` and the student learns the
trace as well as the answer. `--thinking-budget` caps the trace at 1000
tokens and `--max-tokens` gives 320 more for the answer. `--temperature`
and `--top-p` are the sampling settings, how much the teacher varies its
wording and how much of its vocabulary it draws from, and 0.6 with 0.95
gives varied replies that stay on task.

A teacher without a thinking mode runs without `--thinking` and
`--thinking-budget`, since `gen` refuses a budget on its own. The caches
below then take `--frame reply` instead of `reply-think`, and `eval`
under [Use and measure the adapter](#use-and-measure-the-adapter) drops
`--reply-think`. Pair a thinking teacher with a student that has a
thinking mode of its own, or run the teacher without `--thinking` and
`--thinking-budget`.

`filter` runs its checks in a fixed order and names the first one a row
fails, with one reason word per dropped row:

- `length`, the reply did not reach its end of turn.
- `budget`, the reasoning trace hit `--thinking-budget`.
- `empty`, the answer has fewer than `--min-words` words.
- `marker`, a marker of the chat template leaked into the reply or its
  reasoning trace.
- `repeat`, lines or phrases repeat in the reply or its reasoning trace,
  the trace under its own `--max-trace-repeat`.
- `ascii`, too many non-ASCII characters, only with `--max-non-ascii`.
- `tokens`, the reply is over `--max-reply-tokens`.
- `verify`, your checker said no, with its word under `detail` in the
  rejects file.

A reply whose reasoning trace hit the budget is unfinished and is
dropped as `budget`. Expect to lose about a third of the replies at a
1000-token budget, so write more prompts than the rows you need.
`--min-words` counts words of the answer, not of the trace, and its
default of 16 suits prose replies. A right answer here can be one short
query, so the command sets it to 1. The report file has the kept and
dropped counts.

`--max-reply-tokens` counts the trace and the answer together, and its
value depends on your answers. Set it to `--thinking-budget` plus the
longest answer a finished reply needs. A reply over that limit reasoned
for nearly the whole budget or wrote an unusually long answer, and both
are suspect. The worked answers ran under 180 tokens, so the command
sets 1180. Each reply row records its counts under `gen`, and the
one-liner below prints the median and the longest answer among replies
that stayed under the budget. Run it after `gen` and before `filter`:

```sh
python3 -c 'import json,sys; a=sorted(g["completion_tokens"]-g["reasoning_tokens"] for l in open(sys.argv[1]) for g in [json.loads(l)["gen"]] if not g["budget_hit"]); print(a[len(a)//2], a[-1])' r1-replies.jsonl
```

`cache` runs the teacher over the kept replies and ends with a
`[cache] done:` line that says the validator passed. `--frame reply-think`
tells `cache` how to read each row. A frame names which positions are
targets, the tokens the student is trained on. Here the rows are
conversations and the targets start at the final turn's reasoning
trace. `--frame-kwargs` sets the same thinking switch `gen` used, so the
teacher reads the conversation the way it wrote it.

`gen --thinking` sends the server's thinking switch with every request
and starts the teacher with `--thinking on`, so serve maps it onto the
variable the teacher's template reads, `enable_thinking` for Qwen and
the family's own name elsewhere. `cache` renders rows in process, so
`--frame-kwargs` names that variable itself, `{"enable_thinking": true}`
for a Qwen teacher. `align` reads the switch from the cache, so it needs
no flag of its own.

`--max-len` is the longest window, the stretch of a row cached as one
piece, in teacher tokens. A longer text row is cut into windows at word
boundaries. A longer reply row loses its oldest turns first, and is
dropped and counted on the `[cache] frame` line when the last exchange
alone does not fit. 2560 holds a 1180-token reply behind a prompt.
`--top-k` is how many next-token candidates are stored per position,
256 by default, and is unrelated to the sampler's `--top-k` on `gen`.

`align` runs on the CPU with the two tokenizers only and writes the view
into `view-r1/`. It holds back about one row in fifty for validation,
whole documents at a time and at least one row, and the rest are training
rows. A cache made from one document splits that document instead. Its
summary line reports `a` and `s`, the own-group and singleton fractions
explained under [Advanced settings](#advanced-settings), and the other
fields on that line are diagnostics. On the worked pair every teacher
token has a
student token of its own, `a=1.000`, which is the ideal, and a lower
value means a weaker result.

`train` reads the view, fits the adapter on the quantized student, and
writes the adapter GGUF. `--iters` counts steps of `--batch-size` rows
each, and two passes over the rows is enough for a generated corpus. The
one-liner below counts the training rows once `align` has written the
view. Run it after `align` and before `train`. The 3 in it is
`--batch-size`, so change it with the batch size:

```sh
python3 -c 'import json,math,sys; n=sum(e["split"]=="train" for v in sys.argv[1:] for e in json.load(open(v+"/view.json"))["index"]); print(n, 2*math.ceil(n/3))' view-r1/
```

300 is that figure for the 450 training rows of the worked view.
`--lora-rank` is the adapter's capacity, `--lora-alpha` its scale, how
strongly the adapter's change is applied, and `--lr` the peak learning
rate, how far each step moves the adapter.
`--ckpt-dir` is where the run keeps its checkpoints, saved states it
can resume from, `./ckpt` by default. These values come from the worked
task, and [Advanced settings](#advanced-settings) says what each one
changes.

Training prints the loss, the figure it drives down, every ten steps.
If the loss has not fallen by step 40, stop the run and check that the
filter kept the rows you expected and that `align` reported `a=1.000`
or a warning you accepted.

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
under [parameter support](api.md#parameter-support). Adding
`--thinking-budget 1000` to `serve` caps the trace at the length the
student trained with.

The server registers the adapted model under an id derived from the file
name, `qwen3.5-9b` here, and the bare base as `<id>-base`. Both are
listed by `curl localhost:8080/v1/models`, and a request names one of
them:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen3.5-9b",
  "messages": [{"role": "user", "content": "How many manifests have no arrival yet?\n\nAnswer with one SQLite query in a ```sql code block and nothing else."}]
}'
```

Replies carry the answer under `choices[0].message.content` and the
reasoning trace under `reasoning_content` beside it. The question
carries the same closing sentence as every training prompt, since the
student learned to answer in that form. Any OpenAI client library talks
to the same server with `http://127.0.0.1:8080/v1` as its base URL, any
string as the key and the model id above.

Stop the server with `gmlx stop` before the next step, because it
detaches and stays in memory, and `gen`, `cache` and `train` each need
the memory to themselves.

[Use the adapter](lora.md#use-the-adapter) in the LoRA guide covers `run`
and `serve`, and its [interop section](lora.md#adapter-format-and-interop)
says how the same file loads in llama.cpp.

The pass rate uses the same checker that filtered the corpus. Generate
the student's replies to the held-out prompts, without the document, and
run the checker over them:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-heldout.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-r1.jsonl
gmlx distill filter --in heldout-r1.jsonl --out heldout-r1-ok.jsonl \
    --min-words 1 --verify "./check-sql.py freight.sqlite" --report heldout-r1.json
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["kept"]/(r["kept"]+sum(r["dropped"].values())))' heldout-r1.json
```

`--model` names whatever model `gen` serves, here the student with its
adapter, and is the same flag as `--teacher`. `--serve-arg` passes an
argument through to `gmlx serve`, and two of them attach the adapter.
The `=` form is needed because the value starts with `--`. Requests go
to the adapted model, which the server lists first.

The last line prints the pass rate, `kept` divided by `kept` plus the
sum of the `dropped` counts in the report. A reply the budget cut counts
as wrong. Rerun `gen` until it exits 0 before filtering a measurement
set, since a request that failed is a missing row, not a wrong one.

`gen` skips the ids already in its output file, so every measurement
needs its own `--out` and `--report` names. Measure the untouched
student, without the adapter, the same way:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --prompts prompts-heldout.jsonl \
    --thinking --thinking-budget 1000 --max-tokens 320 --temperature 0.6 --top-p 0.95 \
    --out heldout-base.jsonl
gmlx distill filter --in heldout-base.jsonl --out heldout-base-ok.jsonl \
    --min-words 1 --verify "./check-sql.py freight.sqlite" --report heldout-base.json
```

Adding `--context schema.md` to that `gen`, under a new output name,
gives the untouched student with the document pasted in, the figure the
adapter aims for. Run the other two prompt sets through the untouched
student the same way for its column of [What to expect](#what-to-expect). The other two prompt sets run through the adapted
student the same way, each with its own names:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-untrained.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-untrained-r1.jsonl
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-heldout-combined.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-combined-r1.jsonl
```

The first gives the adapter's pass rate on kinds it never trained on,
and the second its pass rate on the combined questions, each after the
same `filter` line with its own `--out` and `--report`.

`eval` scores what a pass rate cannot see. With the census JSON from the
check above it scores the adapter at the positions the document moved,
and it checks that the student's general behavior survived:

```sh
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --before \
    --reply-slice heldout=heldout-ctx.jsonl --reply-think --reply-positions census.json \
    --frame-kwargs '{"enable_thinking": true}' --md eval.md --json eval.json
```

`--before` scores the same loaded model a second time with the adapter
switched off, so both figures come from one process. `--reply-slice`
scores the teacher's held-out replies from the census check on the bare
prompt under `student_messages`, so the document is not in view.
`--reply-think` includes their reasoning trace, and `--reply-positions`
keeps only the positions the document moved. `--frame-kwargs` gives
`eval` the thinking switch, since it has no cache to read one from.
[Read the numbers](#read-the-numbers) explains the report.

`--chat-sanity chat-sanity.jsonl` on that command adds a check that the
student still behaves as a chat model. The file is a jsonl of a few
dozen ordinary prompts in the `messages` form above, each with a `kind`
of `task` or `refuse`, where `refuse` marks a prompt the untouched
student declines. A row reads:

```json
{"id": "cs-001", "kind": "task", "messages": [{"role": "user", "content": "Summarize this paragraph in one sentence: ..."}]}
```

## Round two trains on the student's own right answers

A second round lifts the pass rate a few points over round one. The
student with its first adapter answers the training prompts without the
document, and the checker keeps the right replies. The filter then puts
the document back on the teacher's side, and the teacher is cached over
those replies, so the student is scored against the teacher's view, with
the document, of replies the student wrote:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-train.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r2-replies.jsonl
gmlx distill filter --in r2-replies.jsonl --out r2-corpus.jsonl --min-words 1 \
    --max-reply-tokens 1180 --verify "./check-sql.py freight.sqlite" --context schema.md
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r2-corpus.jsonl --out cache-r2/ \
    --frame reply-think --frame-kwargs '{"enable_thinking": true}' --top-k 256 --max-len 2560
gmlx distill align --cache cache-r2/ --student Qwen3.5-9B-Q6_K.gguf --out view-r2/
gmlx distill train --view view-r1/ --view view-r2/ --student Qwen3.5-9B-Q6_K.gguf \
    --adapter-out r2.gguf --iters 678 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 \
    --ckpt-dir ckpt-r2/
```

`filter --context` rebuilds every kept row with the document in the
teacher's prompt and the bare prompt under `student_messages`. The cache
then holds what the teacher thinks of the student's own words with the
document in view, while the student trains on the prompt alone. A run
whose prompt rows carry their own `context` field has no single document
for `filter --context`, so it stops after round one.

`train` takes both views and starts from the base weights, not from the
first adapter, so `--iters` grows with the rows. The one-liner under
Round one counts them when given both view directories, and 678 is its
figure for the 450 rows of round one plus the 566 of round two.
`--ckpt-dir` keeps the two rounds' checkpoints apart, since both would
write to `./ckpt` without it. Measure `r2.gguf` the same way as round
one, with new output names.

## What each step costs

The figures below are from the worked task, with a 27B teacher at Q8, a
9B student at Q6_K, 615 training questions plus 264 combined ones, and
two rounds. The whole run took about 11 hours, and the census check
before it is one `gen` over the held-out prompts, under an hour, plus
two short caches.

| Step | Memory | Time | Disk |
|---|---|---|---|
| `gen` | the teacher, served, 36 GB for the worked file | 52 teacher tokens per second at `--concurrency` 8, about 4 hours for the training prompts | the replies, a few MB |
| `cache` | the teacher plus a few GB, about 40 GB here | about 500 teacher tokens per second | 1.6 KB per position at top-k 256 |
| `align` | the tokenizers only | about 25 ms per conversation row on the CPU, under 1 ms per plain-text row | a few MB, unless `--materialize` writes the batch tensors out too |
| `train` | 50.7 GB peak on the 9B student | 9.3 s per step of 3 rows, so 678 steps in 1.75 h | two checkpoints under `--ckpt-dir`, each the adapter's parameters plus two running averages of the same size |
| `eval` | the student | minutes per slice, longer with the adapter attached | two report files |

A cache is 6 x K + 22 bytes per position plus the text, with K the
`--top-k` value, so a corpus of 600 rows near 1300 tokens takes about
1.2 GB at the default K. `--max-disk-gb` refuses a cache whose estimate
exceeds it, and every size flag counts decimal GB.

When a run is over, the caches, the checkpoint directories, the server
logs and the sidecars can all go. The views, the adapter and the reports
are what you keep.

The adapter file holds the last step of the run. `best`, the checkpoint
with the lowest validation loss, also stays under `--ckpt-dir`. No
action turns it into an adapter, and `--resume` continues from the last
checkpoint, not from `best`, so a run whose validation loss rose near
the end is rerun with fewer steps.

Training memory scales with the row length and the batch.
`--batch-size 1` and `--grad-checkpoint`, which recomputes intermediate
values during training instead of keeping them, are the two levers when
the student does not fit, each at some cost in time.

A teacher that does not fit is a different problem, and the first answer
is a smaller quantization of it. `cache` refuses a dense teacher over
the wired budget rather than run it resident.

MoE models, mixture-of-experts models, have layers split into experts of
which a few run per token, and
`gmlx validate` says whether a file is one. `cache` streams such a
teacher's experts from disk on its own when the model is larger than
memory. `gen` alone can also use a teacher served elsewhere through
`--base-url`, and a teacher that fits nowhere local cannot be cached, as
[Limitations](#limitations) says.

The document's length is bounded by the teacher's context window, the
most tokens it reads at once, during `gen`, and by `--max-len` during
`cache`, since the document sits in the teacher's prompt on every row.
A few thousand tokens, at roughly four bytes of English text per token,
is comfortable at the settings above. A longer document raises
`--max-len` and the memory of every teacher step with it. Past that, split
the document into sections and give each prompt row the section it needs
under its own `context` field, as [Advanced settings](#advanced-settings)
describes.

## Read the numbers

Four numbers carry most of what the reports print:

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
steps and at the last step. These are the round-one lines at step 200:

```
[train] it 200 loss 0.0465 dk 0.0332 alm 0.0134 ce 0.1682 floored 0 lr 1.43e-05 260 tok/s step 8700 ms load 100 ms peak 49.7 GB active 13.5 cache 8.0
[train] it 200 val 0.0806
```

Every later validation line adds the best earlier value in parentheses.
A validation line that reads `val none` means no validation row held a
scored position, and the best checkpoint stays as it was.
The train lines that print `knobs=` or `blocked attention` are
diagnostics, like the `plan:` line of `cache`. The train line has these
fields.

- `it` is the step number.
- `loss` should fall through the first third of the run and then flatten.
- `dk` and `alm` are its two parts. `dk` is the main term, the distance
  between the student's next-token probabilities and the teacher's stored
  ones, with the probability outside the top-k pooled in one bucket,
  which `--loss bucketed`, the default, selects. `alm` compares whole
  chunks of text. It is 0 when `align` logs `path=identity`, meaning the
  student reads every row as exactly the teacher's tokens. On the worked
  pair the two models share a vocabulary, so `a` is 1.000, but every
  reply row carries `student_messages`, the prompt without the document,
  so the student reads other tokens than the teacher did. `align`
  therefore logged `student render differs`, with the count of rows that
  carry their own student message list, and `path=general`, which is
  expected for any run with a document, and `alm` is nonzero there.
- `ce` is a third term that is measured but not trained on unless `--ce`
  is set.
- `floored` counts positions whose probability was clamped at the
  smallest representable value.
- `lr` is the learning rate at that step.
- `tok/s`, `step` and `load` are the throughput, the wall time per step
  and the time spent reading the batch.
- `peak`, `active` and `cache` are memory in GB, the high-water mark, the
  arrays in use and MLX's buffer cache, memory kept for reuse.

`val` is the loss on a fixed sample of the validation rows, the rows
`align` set aside and `train` never trains on, drawn once across every
view. `best` is the lowest of the earlier validations, so a `val` below
it is a new best, and the first validation line has no `best` yet. A
validation loss that rises while the training loss keeps falling means
the adapter is memorizing the rows, and fewer steps or a lower rank fix
it.

`eval` writes a Markdown report with one table per kind of measurement,
each row a slice, one held-out file. The reply table is the one this
guide reads:

| reply slice | bpb after | bpb before | nats/token after | nats/token before | se | rows |
|---|---|---|---|---|---|---|
| heldout | 0.2225 | 1.4131 | 0.6946 | 4.4116 | 0.0056 | 173 |

`after` is with the adapter and `before` without it, filled in by
`--before`. With `--reply-positions` the rows are scored at the
positions the document moved only, and the log line says so.

Judge the nats per token figure against the teacher's own at the same
positions, the second number, `with`, in the census report's
`teacher nats per token on high-delta positions` row.

`se` is the standard error of `bpb after` over rows, the spread another
sample of rows would show, and two adapters whose bpb differs by less
than two of it cannot be told apart.

The other tables follow the same after and before pattern.

- The slice table scores plain-text slices. `teacher bpb` is filled in
  when `--teacher-bpb` supplies it. `decontam` is the share of a slice's
  64-byte windows found in the training corpus, and a slice over one
  percent has `void` in its `gate` column, meaning the slice was trained
  on and its score does not count.
- The chat slice table scores conversations on their assistant turns.
- The task table gives accuracy on the local task files.
- The chat sanity table gives the share of replies that kept the turn
  structure as `compliance` and the share the budget cut as
  `truncated_rate`. `refusal_rate` is the share of refusal prompts the
  student refused, and `task_refusal_rate` the share of task prompts it
  refused. `ref_nll_nats` is the student's surprise at the replies of an
  earlier report given by `--chat-refs`, or under `--before` at the
  adapter-off replies, which then win. `refs_source` in the JSON names
  which, the report path or `before`.
- The KL table gives the KL divergence, a distance between the student's
  next-token probabilities and the teacher's stored ones, in nats.
  `clustered se` is its standard error over rows, and `top-1` the share
  of positions where both pick the same token.

## What to expect

The worked task, with the teacher and student on one tokenizer, gave
these pass rates.

| pass rate on | untouched student | after round one | after round two |
|---|---|---|---|
| held-out questions of the trained kinds | 0.022 | 0.817 | 0.882 |
| questions of kinds never trained on | 0.017 | 0.917 | 0.925 |
| combined held-out questions | 0.000 | 0.767 | 0.783 |

With the schema pasted into its prompt the untouched student scores
0.946 on the held-out questions, so the adapter reaches most of what
pasting the document gives. At the positions the document moved, the
student's nats per token fell from 4.41 to 0.70, against the teacher's
0.41 with the schema in view.

Whether the student's general behavior survived is measured by the
`eval` flags that score a chat prompt set, a conversation set and local
task files, `--chat-sanity`, `--chat-slice` and `--tasks` under
[distill eval](cli.md#distill-eval). On the worked task none of those
moved outside its noise.

A student from another tokenizer family on the same task closed about a
third of the gap between the untouched student and the student with the
document pasted in. Measure it on your own document before relying on
it.

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
`--frame-instruction`.

`--iters 2000` at the default batch of 8 is two passes over 8000 rows of
512 tokens, about four million teacher tokens, which is a starting size
for general text. Size the corpus by the validation line from there. A
`val` that stops falling while `loss` keeps falling wants more text, and
a `val` still falling at the last step wants more steps.

The two slices are text files you set aside and did not put in
`corpus.jsonl`. `eval --cache` checks each slice against the corpus so a
slice the student trained on is marked, and a same-tokenizer
`--kld-cache` reports the distance from the teacher's stored choices.
Pointing `--kld-cache` at the training cache measures how well the
student fits the text it trained on. A cache made from a held-out file
measures it on new text.

Without a prompt set, `gen --corpus` builds continuation prompts from a
text corpus instead, quoting the start of each document under an
instruction to continue it. It skips documents under `--min-chars`, 2000
characters by default, so a corpus of short documents needs a lower
value or yields no prompts.

A behavior, such as answering in a fixed format, is round one without a
document. Write prompts that call for the behavior, run `gen` without
`--context`, and give `filter` a checker for the format, or no `--verify`
at all when the teacher's replies are the standard. The rest of the
round is unchanged.

## When something goes wrong

The filter dropped most rows. The rejects file names the reason per row.
`budget` means the reasoning trace hit `--thinking-budget`, so raise it
or drop both `--thinking` and `--thinking-budget`. `length` means the
answer hit `--max-tokens`.
`verify` with your own reason words means the teacher got the task wrong
with the document in view, and a teacher that fails most of a task
cannot teach it.

`align` printed a `warn:` line naming the own-group fraction `a` or the
singleton fraction `s`. The two tokenizers split text differently enough
that part of the teacher's output has no direct student target. An `a`
under 0.90 or an `s` under 0.50 means the run works with less signal,
and an `a` under 0.70 refuses and writes no view. Pick a student from
the teacher's family, or pass `--force` and expect a weaker result.

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

The round-one pass rate is low but not zero. Check the census effect
first, since a small one caps what any adapter can learn. Then add
prompts and phrasings for the kinds that fail, including rows that
combine two kinds, and rerun with more steps. A pass rate that stays
low after that wants a larger student.

`gen` refuses because its port has a listener. A `gen` killed without
cleanup leaves its server running, and the refusal names the port. Stop
it with `gmlx stop --port 8093`, or the port you gave `--port`.

`train` runs out of memory. Lower `--batch-size` to 1, add
`--grad-checkpoint`, or cache with a shorter `--max-len` so the rows are
shorter (with a reply frame that drops the replies that no longer fit).

A refused view means the cache it was built from changed, or the
tokenizer tables do not match the student, so rerun `align`. A refusal
naming fewer train rows than `--batch-size` means the corpus is too
small for that batch.

`eval` refuses `--reply-positions` when the census map names none of
the reply rows. The census keys its map by the corpus ids it was given
`--corpus`, so run it with the same corpus file the reply slice was
drawn from. A reply table showing `None` scored no row at all: every
row was dropped as too long for `--chat-max-len`, or had no target
bytes.

Other failures are in [troubleshooting](troubleshooting.md).

## Advanced settings

Each flag here changes one thing, and the full tables are under
[gmlx distill](cli.md#gmlx-distill).

The cache stores, per position, the teacher's `--top-k` most likely next
tokens with their log-probabilities and the log-probability of the token
that followed. It also stores the probability mass outside the top-k,
the probability summed over every other token, and the mass on word
boundaries. `--resume` continues after the last shard it wrote and
verified, and `--validate DIR` checks an existing cache without loading
a model.

Two cache flags serve MoE teachers. A MoE teacher larger than memory
streams its experts from disk. `--routes` stores the experts the teacher
chose, so `eval --kld-cache` on the teacher's own quantization measures
the error the quantization adds and nothing else. `--hidden` stores a
sketch of the teacher's final hidden state, a fixed-width compressed
copy of it, for the `train --hs` term, which is off by default.

A frame says where the targets are in a conversation row. `none`, the
default, caches plain text rows. `chat` targets every assistant turn,
`reply` the final one, `reply-think` the final one from its reasoning
trace onward, and `continue` wraps plain text in an assistant turn.
`--per-turn` makes one row per assistant turn with the history before
it. `--context-format` on `gen` and
`filter` decides how the document and the question combine in the
teacher's prompt, and a prompt row's own `context` field takes
precedence over `gen --context`. That is how several documents share
one run, and how one document too long for the context window is split
into sections.

`align` writes `view.json`, with the row index and the train and
validation split, and `tables.safetensors`, which depends only on the
tokenizer pair. A view is bound to the cache it was built from and to
that pair. `align --tables` points at an earlier view directory whose
tables are reused when the tokenizer pair matches, which saves the build
on a cross-tokenizer pair. A pair with one vocabulary, like the worked
one, rebuilds them in moments and ignores the flag. `align` takes the
identity path when the student renders
every row to the same tokens as the teacher, and the general path
otherwise, which it logs as `path=`.

On a cross-tokenizer pair `align` finds the byte offsets where both
tokenizations agree on a boundary. At each one it maps the teacher's
top-k onto groups of student tokens that start with the same bytes, so a
digit run or a longer merge becomes a target for a sum of student
probabilities. Between boundaries the student is trained to match the
teacher's probability of the whole chunk of bytes. The own-group
fraction, `a` on the `align` summary line, is how much of the teacher's
mass has a direct target. The singleton fraction, `s` on that line, is
how much of it lands on groups of one student token, an exact
one-to-one target. `view.json` also records the shared-boundary
fraction, how much of the text the boundaries cover, and the rest of the
summary line is diagnostics. A mean own-group fraction under 0.90, or a
singleton fraction under 0.50, prints a warning. An own-group fraction
under 0.70 refuses and writes no view unless `--force` is given.

The training loss is a sparse KL over the top-k plus a tail bucket for
the mass outside it, so the student is never asked to put all of its
probability on the top-k. `--dk` weights that term, printed as `dk` in
the train line. `--alm` weights the chunk term of a cross-tokenizer
pair, printed as `alm`. The flag help calls it the chunk term (ALM),
short for approximate likelihood matching. `--ce` weights a plain
cross-entropy on the teacher's tokens, printed as `ce` and left at 0.
`--loss paper`
is the top-k term with no tail bucket, and `--loss renorm` rescales both
distributions to sum to one over the top-k.

`--lora-rank` sets the adapter's capacity, 128 on the 9B student, and
`--lora-alpha` its scale as alpha over rank. `--lr` is the peak learning
rate. The rate rises over the first `--warmup` fraction of the steps and
then falls along a cosine curve to zero. These values come from the
worked task. On a student of another size, start from them and change
one at a time, judged by the validation loss. `--seed` fixes the batch order
and the adapter's initialization, and `--resume` restarts at the exact
step from `--ckpt-dir`. `--view` repeats to train on several views over
one tokenizer pair, aligned with the same chunk settings (`--gamma`,
`--max-chunk-len`, `--w-mid`). The first view's `--T-dk` and `--tau-alm`
apply to all of them unless the train flags override them.

`--iters` follows from the training row count. One pass over the rows,
an epoch, is the train rows of every view divided by `--batch-size`,
rounded up, and the settings above make two passes. The one-liner under
[Round one](#round-one-trains-on-the-teachers-replies) computes it.

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
  `--tokenizer`. It cannot be cached, since `cache` runs the teacher
  itself and needs a local GGUF.
- Recorded routes are replayed by `eval --kld-cache` when the student
  carries the teacher's MoE layers and no adapter is loaded, so a
  requantized teacher is scored on the teacher's own routes and an
  adapter's routing changes count against it. `train` does not replay
  them, so a MoE student trained from a MoE teacher of the same family
  learns from the teacher's outputs alone.
- The hidden-state term reads the teacher's final hidden state only. No
  intermediate layer is stored, and the sketch is fixed at cache time.
- Task files for `eval` are read from disk, in the formats listed under
  [distill eval](cli.md#distill-eval). Nothing is downloaded, so the
  ARC-Easy, HellaSwag and GSM8K files, public multiple-choice and
  arithmetic benchmarks, are yours to fetch and convert. Everything else
  runs offline, apart from `gmlx pull`.
- One cache serves any student, but a view is bound to its cache and its
  tokenizer pair, and `train` refuses one whose cache changed.
- The adapter encodes the document, so sharing the adapter shares the
  document's content.
