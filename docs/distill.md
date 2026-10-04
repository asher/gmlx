# Distillation

Distillation teaches a small model what a larger one knows about a
document. A large model, the teacher, reads the document and answers
questions about it. A small model, the student, trains on those answers.
You get a LoRA adapter, a small GGUF file of extra weights, and with it the
student answers the same questions without the document in its prompt.

```sh
# The teacher answers your prompts with the document in view.
gmlx distill gen --teacher teacher-Q6_K.gguf --prompts prompts-train.jsonl --context schema.md \
    --out replies.jsonl
# Drop unfinished and repeated replies. A checker of your own also drops wrong ones.
gmlx distill filter --in replies.jsonl --out corpus.jsonl
# The teacher scores every token of the replies.
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --frame reply --out cache/
# The scores are mapped to the student's tokens.
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
# The student learns them, and you get an adapter.
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
# Score the student with and without the adapter on text it did not train on.
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --slice prose=heldout-prose.txt --md eval.md --json eval.json
```

Put your own files in place of these names. `schema.md` is the document,
the prompt file holds your questions, in the format that [Write the prompts](distill-walkthrough.md#write-the-prompts)
shows, and `heldout-prose.txt` is text you kept out of training.

In [the worked run](internals/distill.md#the-worked-run), a 9B student
learned a database schema this way. With the adapter it came close to the same
student with the schema pasted into every prompt. The same steps also train
a student on plain text, or on a behavior such as a fixed answer format.

- [What you need](#what-you-need)
- [The actions](#the-actions)
- [A ten-minute smoke run](#a-ten-minute-smoke-run)
- [The steps of a real run](#the-steps-of-a-real-run)
- [Resume a stopped step](#resume-a-stopped-step)
- [Train on plain text or a behavior](#train-on-plain-text-or-a-behavior)
- [Disk use](#disk-use)
- [Limitations](#limitations)

## What you need

- A teacher and a student, both GGUF files. Pick two models from one
  family, such as Qwen3.6 and Qwen3.5, so they share a tokenizer. A student
  from another family works, but learns much less.
- A chat model as the student, since the answers are conversations.
- Enough memory for the teacher, and for the student plus its training
  state. No step loads both at once. On the walkthrough's pair, `train`
  peaked near 51 GB and `cache` near 40 GB, so a 64 GB Mac runs it with
  nothing else large open.
- For a task: the document, a few hundred questions about it, and a script
  that checks an answer.

`gmlx pull --to .` saves a model in the current folder, so that commands
can name it by its file name. The `--teacher` and `--student` flags take
any file path.

## The actions

Every flag is in the [distillation reference](distill-reference.md).

| Action | What it does |
|---|---|
| `gen` | Serves a model and writes its replies to your prompts as a corpus, the rows the student trains on. |
| `filter` | Drops unfinished or repeated replies, and the replies your checker rejects. |
| `cache` | Runs the teacher over the corpus once and stores its likely next tokens at each position. |
| `align` | Maps the cache onto the student's tokenizer and writes a view, the targets the student trains to match. |
| `train` | Fits a LoRA adapter on the student against one or more views. |
| `eval` | Scores the student with and without the adapter on held-out text. |
| `census` | Measures how much the document changes what the teacher says. |

## A ten-minute smoke run

Run the pipeline on a tiny pair before you commit hours to a real one. This
run fits on any Apple Silicon Mac. The teacher is Qwen3 0.6B at Q8_0, and
the student is the same model at Q4_K_M:

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

The run passes when the loss on the `[train] it` lines falls and `eval`
writes `smoke.md`. In that report, `bpb after` (the student with the
adapter) must be below `bpb before`, since the slice is the training text.

`align` holds back about one row in fifty for validation, and `train`
never trains on those rows. `train` needs at least `--batch-size` training
rows, so 24 rows leave 23 for a batch of 4.

`align` also prints `a=` on its summary line: the share of the teacher's
likely tokens that map straight onto student tokens. An `a` of 1.000 is
ideal, and under 0.90 the student learns less. For a cross-family pair, run
this smoke corpus through `cache` with your teacher and `align` with your
student to see the figure before a real run.

Then check `gen` and `filter` on the same pair:

```sh
python3 -c 'import json; [print(json.dumps({"id": f"smoke-{i}", "messages": [{"role": "user", "content": f"Describe lighthouse number {i} in one sentence."}]})) for i in range(2)]' > smoke-prompts.jsonl
gmlx distill gen --model Qwen3-0.6B-Q8_0.gguf --prompts smoke-prompts.jsonl --max-tokens 96 \
    --out smoke-replies.jsonl
gmlx distill filter --in smoke-replies.jsonl --out smoke-corpus.jsonl --min-words 1
```

Both pass when `gen` ends with a `[gen] done:` line that reports `0 failed`
and `filter` prints `[filter] kept 2`. `--model` is another name for
`--teacher`. `--min-words 1` keeps one-sentence replies, which the default
of 16 words drops.

## The steps of a real run

A run on your own document takes these steps. The
[walkthrough](distill-walkthrough.md) shows each one with its commands.

1. Write [the prompts](distill-walkthrough.md#write-the-prompts) and
   [the checker](distill-walkthrough.md#write-the-checker) for your document.
2. [Check that the document matters](distill-walkthrough.md#check-that-the-document-matters)
   before you spend hours on training.
3. [Round one](distill-walkthrough.md#round-one): the teacher answers with
   the document in view, and the student trains on the right answers.
4. [Serve](distill-walkthrough.md#serve-the-adapter) the adapter, and
   [measure](distill-walkthrough.md#measure-the-adapter) it on questions it
   did not train on.
5. [Round two](distill-walkthrough.md#round-two), optional: the student
   trains on its own right answers, scored by the teacher.

The walkthrough's two rounds took about 11 hours, most of it in `gen` and
`train`. [Distillation troubleshooting](distill-troubleshooting.md) explains
the figures each step prints and what to do when one looks wrong.

## Resume a stopped step

Each step that runs for hours can continue where it stopped:

- `gen`: run the same command again. It skips the prompt ids already in
  its `--out` file, so a new run, such as each measurement, needs its own
  `--out` name.
- `cache`: add `--resume`. It continues after the last shard it wrote.
- `train`: add `--resume`. It continues from the last checkpoint in
  `--ckpt-dir`, `./ckpt` by default.

A step refuses to resume when its settings or inputs changed since the
first run, and says so. A new `train` run also refuses a `--ckpt-dir`
that already holds checkpoints, so give each run its own directory.

## Train on plain text or a behavior

To make a small model behave like a larger one on ordinary text, cache the
teacher over the text. This needs neither `gen` nor `filter`:

```sh
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --out cache/ \
    --top-k 256 --max-len 512 --max-disk-gb 20
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --cache cache/ --slice prose=heldout-prose.txt --slice code=heldout-code.txt \
    --kld-cache cache/ --md eval.md --json eval.json
```

The corpus is one of these:

- A jsonl file with a `text` field in each row.
- A folder. Each `.txt`, `.md`, `.py` or `.json` file in it or its
  subfolders is one document, and each line of a `.jsonl` file is one.
  `cache` skips files with other extensions without a warning.
- A Hugging Face dataset id, which needs the `datasets` package.

The two slices are text files you kept out of the corpus.
`eval --cache` marks a slice that overlaps the corpus, so its score does
not count. `--kld-cache` adds a table of how far the student is from the
teacher's stored choices. It needs a teacher and student on the same
tokenizer.

`--iters 2000` at the default batch of 8 is two passes over 8000 rows of
512 tokens, a starting size for general text. When `val` stops falling
while `loss` keeps falling, add text. When `val` still falls at the last
step, add steps.

A behavior, such as a fixed answer format, is
[round one](distill-walkthrough.md#round-one) without a document. Write
prompts that call for the behavior, run `gen` without `--context`, and give
`filter` a checker for the format.

## Disk use

A cache takes 6 x K + 22 bytes per position plus the text, where K is the
`--top-k` value, 256 by default. A corpus of 600 rows of about 1300 tokens
takes about 1.2 GB. `--max-disk-gb` refuses a cache whose estimate is
larger.

One cache serves any student, since `align` maps it onto each student's
tokenizer. A view reads its cache on every `train`, so keep the cache as
long as you use the view. When a run is done, keep the adapter and the reports. You can
delete the checkpoint folders and the `.server.log` files that `gen` writes.

## Limitations

- The student is a GGUF file, and training writes a
  [LoRA adapter](lora.md) for it. An adapter works only with the exact
  file it was trained on.
- Each training run starts from the base weights. A changed document needs
  a new round one, not a top-up of the old adapter.
- `gen --base-url` can use a teacher on another server, but `cache` runs
  the teacher itself and needs a local GGUF.
- `eval` reads its benchmark task files from disk and downloads nothing.
- The adapter encodes the document. Sharing the adapter shares the
  document's content.
