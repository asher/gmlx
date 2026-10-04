# Distillation walkthrough

This page runs one task end to end. A Qwen3.5-9B student learns a database
schema from a Qwen3.6-27B teacher, so that it answers questions with SQL
without the schema in its prompt. gmlx does not ship the task's files, so
read their names as placeholders for your own. Do the
[smoke run](distill.md#a-ten-minute-smoke-run) first.

- [The task](#the-task)
- [Write the prompts](#write-the-prompts)
- [Write the checker](#write-the-checker)
- [Check that the document matters](#check-that-the-document-matters)
- [Round one](#round-one)
- [Serve the adapter](#serve-the-adapter)
- [Measure the adapter](#measure-the-adapter)
- [Round two](#round-two)

## The task

The document is `schema.md`, about 3400 bytes, and only the teacher reads
it. A reply is right when its query runs against `freight.sqlite` and
returns the reference rows. An API reference with tests, or a style guide
with a linter, fits the same steps.

Get the two models:

```sh
gmlx pull hf:unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf --to .
gmlx pull hf:unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q6_K.gguf --to .
```

## Write the prompts

A prompt file is jsonl, one question per line. Each row has a unique `id`
and a `messages` list that ends on a user turn. Other fields travel with
the row unchanged, so the checker finds the reference query under `check`:

```json
{"id": "train-00000", "family": "in_transit_hull",
 "messages": [{"role": "user", "content": "For Kestrel-class ships, how many manifests have no arrival yet?\n\nAnswer with one SQLite query in a ```sql code block and nothing else."}],
 "check": {"sql": "SELECT COUNT(*) FROM manifests m JOIN haulers h ON m.hauler_id = h.hauler_id WHERE h.hull_class = 'Kestrel' AND m.arrived_at IS NULL", "ordered": false}}
```

Write four sets, one to train on and three to measure with:

| File | What it holds |
|---|---|
| `prompts-train.jsonl` | Every kind of question the document answers, three or more phrasings each, plus rows that combine two kinds |
| `prompts-heldout.jsonl` | The same kinds in words the training set does not use |
| `prompts-untrained.jsonl` | Kinds the training set never covers, to see whether the student learned the document or only the questions |
| `prompts-heldout-combined.jsonl` | Held-out questions that combine two kinds, the hardest set |

Every set carries `check`, the measurement sets too. A pass rate runs the
checker on their replies, and a row without `check` counts as `bad_sql`.

The walkthrough trained on 615 rows plus 264 combined ones. Two hundred
held-out rows make a pass rate stable to a few points.

Combined rows, such as a count over a join, matter. Without them the
student answers each kind alone and fails the combinations.

Hundreds of prompts are easiest to write with a script. This skeleton
fills one kind of question, in three phrasings, from the database's own
values, and writes the reference query under `check`. Add an entry to
`shapes` for each kind, including combined ones. Save it as
`make-prompts.py` next to `freight.sqlite`:

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

```sh
python3 make-prompts.py > prompts-train.jsonl
```

For the held-out sets, run it again with new phrasings and the id prefix
`heldout-`.

## Write the checker

The checker reads reply rows as jsonl on stdin and prints one line per row:
`ok`, or a word that says why the reply is wrong. It must exit 0. Save this
as `check-sql.py` and run `chmod +x check-sql.py`:

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

It opens the database read-only, so a reply that deletes rows does no harm.

For a task with no mechanical check, the checker can ask a served model
whether the reply matches a reference answer that you wrote. `filter` runs
after `gen` has stopped its own server, so start the judge model with
`gmlx serve` before `filter`, and run `gmlx stop` after it.

## Check that the document matters

Before you train, measure whether the document changes what the teacher
says. `gen --context` puts the document in the teacher's prompt and also
keeps the bare prompt under `student_messages`. `census` compares the
teacher's view of the same replies with and without the document:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-heldout.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-ctx.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus heldout-ctx.jsonl --out cache-heldout-ctx/ \
    --frame reply-think --max-len 2560
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus heldout-ctx.jsonl --out cache-heldout-bare/ \
    --messages-key student_messages --frame reply-think --max-len 2560
gmlx distill census --without cache-heldout-bare/ --with cache-heldout-ctx/ \
    --corpus heldout-ctx.jsonl --out census.json --md census.md
```

Two rows of `census.md` decide whether to go on:

- `distillable effect`: how far the document moves the teacher's choices,
  in nats. Under about 0.05, the run is not worth its hours.
- `high-delta positions`: the share of tokens that the document makes more
  likely by over one nat. Under 0.01, the run is not worth its hours.

Keep `census.json`, since `eval` reads it later. Then check the teacher's
own pass rate on the same replies. A teacher that fails most questions with
the document in view cannot teach them:

```sh
gmlx distill filter --in heldout-ctx.jsonl --out heldout-ctx-ok.jsonl \
    --min-words 1 --verify "./check-sql.py freight.sqlite" --report heldout-ctx-filter.json
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["kept"]/(r["kept"]+sum(r["dropped"].values())))' heldout-ctx-filter.json
```

The last line prints the pass rate: kept rows over all rows.

## Round one

The teacher answers the training prompts with the document in view, the
checker keeps the right replies, and the student trains on them. Run the
commands one at a time, since two values depend on your data:

```sh
gmlx distill gen --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --prompts prompts-train.jsonl \
    --context schema.md --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r1-replies.jsonl
gmlx distill filter --in r1-replies.jsonl --out r1-corpus.jsonl \
    --min-words 1 --max-reply-tokens 1180 --verify "./check-sql.py freight.sqlite" \
    --report r1-filter.json --rejects r1-rejects.jsonl
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r1-corpus.jsonl --out cache-r1/ \
    --frame reply-think --top-k 256 --max-len 2560
gmlx distill align --cache cache-r1/ --student Qwen3.5-9B-Q6_K.gguf --out view-r1/
gmlx distill train --view view-r1/ --student Qwen3.5-9B-Q6_K.gguf --adapter-out r1.gguf \
    --iters 300 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 --ckpt-dir ckpt-r1/
```

`gen` turns on the teacher's thinking, caps the reasoning at 1000 tokens,
and gives the answer 320 more. The student learns the reasoning as well as
the answer. For a teacher without a thinking mode, change the recipe on this
page:

- Drop `--thinking` and `--thinking-budget` from every `gen`.
- Use `--frame reply` instead of `reply-think` on `cache`.
- Drop `--reply-think` and `--frame-kwargs` from `eval`. `--frame-kwargs`
  passes the student template's thinking switch, `enable_thinking` on Qwen.
- Serve the adapter without `--thinking on`.

`filter` drops a reply whose reasoning hit the budget, about a third of
them at 1000 tokens, so write more prompts than the rows you need. Set
`--max-reply-tokens` to the budget plus the longest finished answer. This
prints the median and longest answer, in tokens, after `gen`:

```sh
python3 -c 'import json,sys; a=sorted(g["completion_tokens"]-g["reasoning_tokens"] for l in open(sys.argv[1]) for g in [json.loads(l)["gen"]] if not g["budget_hit"]); print(a[len(a)//2], a[-1])' r1-replies.jsonl
```

`--frame reply-think` trains the student on the final turn from its
reasoning onward, and `--max-len 2560` holds a 1180-token reply behind its
prompt. `align` runs on the CPU, and on this pair its summary line reads
`a=1.000`, the ideal.

Set `--iters` to two passes over the training rows. This counts the rows
after `align` and prints the step count for a batch of 3. For this view it
printed 450 rows and 300 steps:

```sh
python3 -c 'import json,math,sys; n=sum(e["split"]=="train" for v in sys.argv[1:] for e in json.load(open(v+"/view.json"))["index"]); print(n, 2*math.ceil(n/3))' view-r1/
```

The rank, alpha and learning rate suit this 9B student. The defaults are
rank 16, a batch of 8 and a learning rate of 1e-4. The loss on the
`[train] it` lines should fall by step 40, or see
[The loss does not fall](distill-troubleshooting.md#the-loss-does-not-fall).

## Serve the adapter

The student trained with reasoning, so serve it with thinking on:

```sh
gmlx serve Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --thinking on
```

The server lists the adapted model as `qwen3.5-9b` and the bare student as
`qwen3.5-9b-base`. Ask in the same form as the training prompts, with the
same closing sentence. [Use the adapter](lora.md#use-the-adapter) covers
`gmlx run` and other clients.

Run `gmlx stop` before the next step. `gen`, `cache` and `train` each need
the memory to themselves.

## Measure the adapter

A pass rate is the share of held-out questions the checker accepts. This
recipe measures the adapter on the held-out set. `gen` serves the student,
and the two `--serve-arg` flags attach the adapter. The first needs the `=`
form, since its value starts with `--`:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-heldout.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out heldout-r1.jsonl
gmlx distill filter --in heldout-r1.jsonl --out heldout-r1-ok.jsonl \
    --min-words 1 --verify "./check-sql.py freight.sqlite" --report heldout-r1.json
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["kept"]/(r["kept"]+sum(r["dropped"].values())))' heldout-r1.json
```

Rerun `gen` until it exits 0 before you filter, since a failed request is a
missing row, not a wrong one. Run the recipe once per row of this table,
each with its own `--out` and `--report` names:

| Measures | Change to the recipe |
|---|---|
| The adapter on held-out questions | None |
| The untouched student | Drop both `--serve-arg` flags |
| The target: the student with the document pasted in | Drop both `--serve-arg` flags, add `--context schema.md` |
| Kinds never trained on | `--prompts prompts-untrained.jsonl` |
| Combined questions | `--prompts prompts-heldout-combined.jsonl` |

`eval` measures what a pass rate cannot see. With the census JSON, it
scores the student on the tokens the document moved, with the document out
of view:

```sh
gmlx distill eval --student Qwen3.5-9B-Q6_K.gguf --adapter r1.gguf --before \
    --reply-slice heldout=heldout-ctx.jsonl --reply-think --reply-positions census.json \
    --chat-max-len 2560 --frame-kwargs '{"enable_thinking": true}' --md eval.md --json eval.json
```

`--before` also scores the student with the adapter off. Add
`--chat-sanity` with a few dozen ordinary prompts to check that the student
still behaves as a chat model. [The eval report](distill-troubleshooting.md#the-eval-report)
explains the tables, and [the worked run](internals/distill.md#the-worked-run)
has this task's results.

## Round two

A second round adds a few points. The student with its first adapter
answers the training prompts, and the checker keeps its right replies.
`filter --context` puts the document back on the teacher's side, so the
teacher scores the student's own words with the document in view:

```sh
gmlx distill gen --model Qwen3.5-9B-Q6_K.gguf --serve-arg=--adapter --serve-arg=r1.gguf \
    --prompts prompts-train.jsonl --thinking --thinking-budget 1000 --max-tokens 320 \
    --temperature 0.6 --top-p 0.95 --out r2-replies.jsonl
gmlx distill filter --in r2-replies.jsonl --out r2-corpus.jsonl --min-words 1 \
    --max-reply-tokens 1180 --verify "./check-sql.py freight.sqlite" --context schema.md
gmlx distill cache --teacher Qwen3.6-27B-UD-Q8_K_XL.gguf --corpus r2-corpus.jsonl --out cache-r2/ \
    --frame reply-think --top-k 256 --max-len 2560
gmlx distill align --cache cache-r2/ --student Qwen3.5-9B-Q6_K.gguf --out view-r2/
gmlx distill train --view view-r1/ --view view-r2/ --student Qwen3.5-9B-Q6_K.gguf \
    --adapter-out r2.gguf --iters 678 --batch-size 3 --lora-rank 128 --lora-alpha 64 --lr 5e-5 \
    --ckpt-dir ckpt-r2/
```

`train` takes both views and starts again from the base weights, so
`--iters` grows with the rows. Give the row-count one-liner both view
directories. Here it counted 1016 rows and 678 steps. Measure `r2.gguf`
with the same recipe and new output names.
