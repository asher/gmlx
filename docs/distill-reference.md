# Distillation reference

This page lists the flags of every `gmlx distill` action. The walkthrough is
[Distillation](distill.md).


`gmlx distill` trains a LoRA adapter for a small GGUF on a larger one's
outputs in six actions plus one check. The teacher and the student may
use different tokenizers, and the walkthrough is [Distillation](distill.md).

- `gen` runs a teacher through `gmlx serve` over a prompt set and writes
  its replies as a corpus.
- `filter` drops the generated rows a student should not learn from.
- `cache` runs a teacher GGUF over a corpus once and stores its most
  likely next tokens and their log-probabilities at every position.
- `align` maps that cache onto a student tokenizer and writes a view, the
  positions and values the student trains to match.
- `train` fits a LoRA adapter on a GGUF student against the view.
- `eval` scores the student with and without the adapter.
- `census` compares two reply caches of the same replies, one made with a
  context the student never sees, and measures how much that context moves
  the teacher.

```sh
gmlx distill gen --teacher teacher-Q6_K.gguf --prompts prompts.jsonl --out replies.jsonl
gmlx distill filter --in replies.jsonl --out corpus.jsonl
gmlx distill cache --teacher teacher-Q6_K.gguf --corpus corpus.jsonl --frame reply --out cache/
gmlx distill align --cache cache/ --student student-Q4_K_M.gguf --out view/
gmlx distill train --view view/ --student student-Q4_K_M.gguf --adapter-out student-distill.gguf --iters 2000
gmlx distill eval --student student-Q4_K_M.gguf --adapter student-distill.gguf --before \
    --slice prose=heldout.txt --md eval.md --json eval.json
```

Every size flag is in decimal GB, 1e9 bytes. Each action checks its
output paths before any model loads. It exits 0 on success, and 2 on a
refused input or setting, a missing required flag or input file, or an
output path it cannot write. Some actions add codes of their own:

- `gen` exits 1 when some requests failed and their prompts remain to be
  rerun, and 2 when its server fails to start.
- `filter` exits 2 when its `--verify` command fails.
- `cache` exits 2 when a shard cannot be written, and keeps the verified
  shards. It exits 3 when its memory probe misses twice or a `--routes`
  recording does not match its rows, and 4 when the validator fails on
  what it wrote. `cache --validate` exits 1 on a problem.
- `align` exits 3 when the own-group check refuses the pair, and writes no
  view.

### distill gen

`gen` reads a prompt file with one `{"id", "messages", "context"}` object
per line, whose messages end on a user turn. A row's context, or the file given by
`--context`, goes in front of the last user turn for the teacher, and
either one must hold text. A row that took a context is written with the
teacher's list under `messages` and the prompt as given under
`student_messages`. A row without one carries `messages` alone.

Prompt ids already in the output are skipped, so a run resumes where it
stopped. A resume checks that each skipped id still names the prompt it
answered and refuses when one differs or is gone, since ids taken from
line numbers shift when a line is inserted. An interrupt cancels the
queued requests and stops the server, and the run ends when the requests
in flight have failed or returned.

Beside the output, `<out>.gen.json` holds the settings a resume must
match and is written before the first request. When a run ends, the
sidecar gains a `run` block with the reply and token totals read from the output
rows, the wall time summed over the runs that ended, and this run's
failed requests and aggregate token rate. An interrupted run leaves the
block as it found it, and a rerun that finds every prompt answered
writes the block from the rows when the sidecar has none.

A `--base-url` server that lists several models serves the run with the
one named like `--teacher`, and gen refuses when none or several match.

These flags control `gmlx distill gen`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--out PATH` | Required | Write the corpus jsonl here, with `<out>.gen.json` beside it. |
| `--prompts PATH` | None | Read prompt rows that end on a user turn from this jsonl. |
| `--corpus PATH_OR_ID` | None | Build continuation prompts from this text corpus instead of `--prompts`. |
| `--teacher GGUF` | None | Serve this GGUF for the run, the teacher or, for a measurement, the student. `--model` is the same flag. |
| `--base-url URL` | None | Use this running server's `/v1` base instead of serving `--teacher`. With `--thinking-budget` the close is sized for a server that runs a drafter. |
| `--host HOST` | `127.0.0.1` | Bind the served teacher to this host. |
| `--port N` | `8093` | Serve the teacher on this port. |
| `--text-key KEY` | `text` | With `--corpus`, read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | With `--corpus`, read this split of a Hugging Face dataset. |
| `--prefix-chars N` | `1500` | With `--corpus`, quote this many characters of each document in the user turn, cut at a space. |
| `--min-chars N` | `2000` | With `--corpus`, skip documents shorter than this. |
| `--docs N` | All | With `--corpus`, build this many prompts. |
| `--instruction TEXT` | `Continue the following text.` | With `--corpus`, place this user turn before the prefix. |
| `--chat-template-kwargs JSON` | None | Pass this JSON object of template variables to every teacher render, through serve's `--chat-template-config`. A thinking key is refused. |
| `--context FILE` | None | The teacher reads this text for every prompt without its own context field. A blank file is refused. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format, which must place both fields. |
| `--thinking` | Off | Turn thinking on and keep the reasoning trace as `reasoning_content` on the reply. Without it, gen turns thinking off. |
| `--thinking-budget N` | None | With `--thinking`, cap the reasoning trace at N tokens per request, and mark the replies it cut for `filter`. |
| `--tokenizer GGUF_OR_DIR` | `--teacher` | Count the reasoning trace against the budget with this tokenizer when `--base-url` is given. |
| `--serve-arg ARG` | None | Pass this argument to `gmlx serve`. The flag repeats, and a resume checks it. Flags that alter the prompt or thinking, and a drafter with a budget, are refused. |
| `--startup-timeout S` | `900` | Wait this many seconds for the served teacher. |
| `--concurrency N` | `8` | Keep this many requests in flight. |
| `--max-tokens N` | `1024` | Give each request this answer budget. With `--thinking-budget` the trace gets its own budget plus the forced close. Without it, the trace shares this budget. |
| `--temperature F` | `0.7` | Set the sampling temperature. |
| `--top-p F` | `0.9` | Keep the most likely tokens whose probabilities add to this. |
| `--top-k N` | The server's | Keep this many candidate tokens. |
| `--min-p F` | The server's | Drop tokens less likely than this share of the best token. |
| `--seed N` | `1` | Use this base seed, to which each request adds its prompt index. |
| `--timeout S` | `1800` | Fail a request after this many seconds. |
| `--report-every N` | `50` | Print a progress line every N replies. |

`--serve-arg` refuses `--thinking`, `--thinking-budget`, `--chat-template`,
`--chat-template-config`, `--reasoning-effort`, `--system-prompt` and
`--profile`, in any spelling serve accepts. Each changes what the teacher
is prompted with, and the rows would not record it. `--native-mtp`,
`--speculative` and `--draft-gguf` are refused beside `--thinking-budget`
too, because a server that runs a drafter does not hold each request to the
budget.

Set the thinking switch and budget with gen's own flags, template variables
with `--chat-template-kwargs`, sampling with gen's sampling flags, and a
system prompt as a system turn in the prompt rows. A template override has
no gen form, since `cache` renders the rows with the teacher's own
template.

### distill filter

`filter` runs its checks in a fixed order, and the first failure names the
reason. The reason is one of `length`, `budget`, `empty`, `marker`,
`repeat`, `ascii`, `tokens` and `verify`, each defined in
[Round one trains on the teacher's replies](distill.md#round-one-trains-on-the-teachers-replies)
in the distillation guide.

`--context` rebuilds every kept row with the context on the teacher's side
and the prompt as given under `student_messages`. The rebuilt rows prepare
a second round from replies a student wrote without the context. The flag
refuses a row that already carries `student_messages`, since that row was
generated with a context.

These flags control `gmlx distill filter`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--in PATH` | Required | Read this generated corpus jsonl. The flag repeats, and inputs join in order. Sidecars that disagree, inputs filtered differently or shared row ids are refused. |
| `--out PATH` | Required | Write the filtered corpus here, with `<out>.gen.json` beside it. |
| `--report JSON` | None | Write the kept and dropped counts here. |
| `--rejects PATH` | None | Write one `{id, reason}` line per dropped row here, with the checker's word under `detail`. |
| `--min-words N` | `16` | Drop replies whose answer, trace excluded, has fewer than this many units. A unit is a word or one ideograph or kana character. `--min-tokens` is the same flag. |
| `--ngram N` | `8` | The repetition check uses n-grams of this size, in the units of `--min-words`. |
| `--max-repeat F` | `0.2` | Drop replies whose repeated n-grams exceed this fraction. |
| `--max-trace-repeat F` | `0.5` | Drop replies whose reasoning trace's repeated n-grams exceed this fraction. |
| `--max-line-repeats N` | `2` | Drop replies with a line repeated more than this many times in a row, skipping lines without a letter or digit. |
| `--max-non-ascii F` | Off | Drop replies whose non-ASCII character fraction exceeds this. |
| `--max-reply-tokens N` | Off | Drop replies longer than this many tokens, reasoning trace included. |
| `--keep-budget-hit` | Off | Keep replies whose thinking budget cut the reasoning trace. |
| `--verify CMD` | None | Run this shell command as your checker, which reads the rows that passed the earlier checks as jsonl on stdin and prints `ok` or a reason word per row. |
| `--context FILE` | None | Put this text on the teacher's side of every kept row. A blank file is refused. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format, which must place both fields. |

### distill cache

`cache` runs the teacher pass. `--messages-key` picks which list of a row
the teacher reads, and `--student-messages-key` only names the list the
student's render reads later, in `align` and `eval`.

Reply and reply-think rows whose final turn has no content, such as a
tool call, have nothing to target and are dropped, counted in the `[cache] frame` line. That line also counts the
reply-think rows whose reasoning trace the teacher's template does not
render, which train on the reply alone, and a reply-think pass in which
no row keeps its trace is refused.

A corpus written by `gen` renders with the thinking switch and the
`--chat-template-kwargs` its `.gen.json` sidecar records, mapped onto the
variables the teacher's template reads. `--frame-kwargs` adds to them,
and a value that contradicts one is refused. A template that prints the
date, as Llama 3's and gpt-oss's do, renders the day the cache was first
started, and a resume and the student's render in `align` keep that day.

These flags control `gmlx distill cache`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--teacher GGUF` | Required unless `--validate` | Run this teacher GGUF, which may be sharded. |
| `--corpus PATH_OR_ID` | Required unless `--validate` | Read this jsonl file, directory of text files or Hugging Face dataset id, `id[@config]`. |
| `--out DIR` | Required unless `--validate` | Write the cache to this directory. |
| `--validate DIR` | None | Validate an existing cache and exit without loading a teacher. |
| `--top-k N` | `256` | Keep this many log-probabilities per position. |
| `--max-len N` | `2048` | Fill each window with this many teacher tokens, start token included, at least 2. The continue frame and closing tail count toward it and must leave at least 8. |
| `--max-disk-gb F` | None | Refuse when the size estimate exceeds this. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache at this many GB during the pass. |
| `--logits-cap-gb F` | `4.0` | Size the head sub-chunk to fit this memory cap. |
| `--floor` | Off | Also store `floor_kld`, the KL against the f16-rounded top-k. |
| `--rows-per-shard N` | `64` | Write this many rows per shard file. |
| `--trunk N` | `512`, or `8192` streaming | Run the trunk in chunks of this many tokens, with rows stacked on the batch axis. |
| `--resume` | Off | Continue after the last verified shard, refused when the corpus, teacher, template, HF source or row options changed. A finished cache is validated, not redone. |
| `--max-rows N` | None | Stop after this many rows. |
| `--max-tokens N` | None | Stop after this many teacher tokens. |
| `--limit-docs N` | None | Read at most this many documents. |
| `--text-key KEY` | `text` | Read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | Read this split of a Hugging Face dataset id. |
| `--source TAG` | `human`, or `synthetic` with a generator sidecar | Write this source tag on every row. |
| `--frame KIND` | `none` | Place the targets in the chat template by frame, `none`, `continue`, `chat`, `reply` or `reply-think`. `reply-think` starts at the final turn's reasoning trace. |
| `--per-turn` | Off | With the chat or reply frame, write one reply row per assistant turn. |
| `--student-messages-key KEY` | `student_messages` | Name the corpus key that holds the student's own message list on reply rows. |
| `--frame-instruction TEXT` | `Continue the following text.` | Use this user turn for the continue frame. |
| `--messages-key KEY` | `messages` | Read the conversation from this column for the chat and reply frames. |
| `--close-final-windows` | Off | With the continue frame, close the last window of a document with the turn-end marker. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs, an object or a file, to every teacher render, beside those a `gen` sidecar records. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--no-require-feeder` | Off | Run a streaming teacher without the prefill feeder. |
| `--no-wired-limit` | Off | Leave the wired limit where it is for a teacher that fits in memory. |
| `--stream-experts` | Off | Force expert streaming on a MoE teacher that would fit in memory. |
| `--expert-bytes-gb F` | The streamed expert bytes | Report this many expert bytes read per forward pass in the read-traffic report. Use `0` for a resident teacher. |
| `--routes` | Off | On a MoE teacher, store every layer's top-k expert ids per position for replay by `eval`. A gate that cannot replay is refused. |
| `--hidden` | Off | Also store a seeded random sketch of the teacher's final hidden state per position, for `train --hs`. |
| `--hidden-dim N` | `256` | Set the width of the hidden sketch. |
| `--hidden-seed N` | `1` | Seed the sketch matrix. |
| `--cpu` | Off | Run on the CPU device. |

### distill align

`align` maps a teacher cache onto the student's tokenizer and writes the
view that `train` reads.

These flags control `gmlx distill align`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--cache DIR` | Required | Read this cache directory. |
| `--student GGUF_OR_DIR` | Required | Align to this student GGUF, or to an MLX checkpoint directory for its tokenizer. |
| `--out DIR` | Required | Write the view to this directory. An earlier view there is replaced once every check has passed. |
| `--tables DIR` | None | Reuse the tokenizer tables of this earlier view directory when the pair matches. |
| `--kprime N` | The maximum seen | Keep at most this many distinct student-token groups per boundary. The identity path ignores it, since K' = K there. |
| `--materialize` | Off | Also write the batch tensors as view shards. |
| `--max-disk-gb F` | None | Refuse to materialize past this size. |
| `--force` | Off | Keep a view the own-group check would refuse. |
| `--val-fraction F` | `0.02` | Hold this fraction of rows for validation, whole documents at a time. A cache of two or more rows holds at least one. |
| `--seed N` | `1` | Seed the validation split. |
| `--w-mid F` | `0.5` | Weight an intra-word shared boundary by this much. |
| `--gamma F` | `0.001` | In the chunk term (ALM), drop chunks whose teacher boundary mass is under this positive value. |
| `--tau-alm F` | `1.0` | Set the positive temperature of the chunk term (ALM). |
| `--T-dk F` | `1.0` | Set the positive temperature that the KL term's group softmaxes use, under every `--loss` form. |
| `--max-chunk-len N` | `8` | Cap ALM chunks at this many tokens on either side, at least 1. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs to every student render and store them in the view, over those the cache recorded and its `gen` thinking switch. |
| `--cpu` | Off | Run on the CPU device. |

### distill train

`train` fits the LoRA adapter against one or more views and saves
checkpoints as it goes.

These flags control `gmlx distill train`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--view DIR` | Required | Train on this view directory. Repeat the flag to mix views aligned alike over one tokenizer pair. |
| `--student GGUF` | Required | Train this student GGUF, which may be sharded. |
| `--adapter-out PATH` | Required | Write the GGUF adapter here. An unwritable path is refused before the load, and a module the adapter cannot hold before the first step. |
| `--iters N` | Required | Train for this many steps. |
| `--lora-rank N` | `16` | Set the LoRA rank. |
| `--lora-scale F` | `2.0` | Apply this nonzero LoRA multiplier directly. |
| `--lora-alpha F` | None | Set the nonzero LoRA multiplier as alpha over rank, instead of `--lora-scale`. |
| `--lora-dropout F` | `0.0` | Set the LoRA dropout, below 1, with one mask per step that `--grad-checkpoint` replays. |
| `--grad-checkpoint` | Off | Recompute each layer's activations in the backward pass. It is refused on Kimi K3 and DeepSeek-V4.1. |
| `--lr F` | `1e-4` | Set the peak learning rate. |
| `--batch-size N` | `8` | Train on this many rows per step. |
| `--warmup F` | `0.05` | Warm up for this fraction of the steps, at least one step and never the last, then decay by cosine. `0` starts at the peak rate. |
| `--weight-decay F` | `0` | Set the AdamW weight decay. |
| `--clip F` | `1.0` | Clip the gradient norm at this value. `0` turns clipping off. |
| `--seed N` | `1` | Seed the data order and the LoRA init. |
| `--loss MODE` | `bucketed` | Pick the sparse KL variant, `bucketed`, `paper` or `renorm`. |
| `--dk F` | `1` | Weight the bucketed KL term by this much. |
| `--alm F` | `1`, `0` when `align` took the identity path | Weight the chunk term (ALM) by this much. |
| `--ce F` | `0` | Weight the cross-entropy term by this much. |
| `--T-dk F` | The view's | Override the view's T_dk. |
| `--tau-alm F` | The view's | Override the view's tau_alm. |
| `--gamma F` | The view's | Override the view's gamma, refused when it differs on a materialized view (its chunks are cut by `align`). |
| `--chunk N` | `512` | Run the head in chunks of this many positions. |
| `--hs F` | `0` | Weight the hidden-state term, a learned map from the student's final hidden state to the cache's sketch at every boundary. |
| `--hs-loss MODE` | `cosine` | Compare hidden states by `cosine`, or by `mse` on unit vectors. |
| `--ckpt-dir DIR` | `./ckpt` | Write checkpoints to this directory. A fresh run refuses one that holds an earlier run's checkpoints. |
| `--resume` | Off | Resume from `--ckpt-dir`, refused when none exists or when the views, student, training settings or gmlx's validation leave-out rule differ from that run. |
| `--save-every N` | `200` | Save a checkpoint every N steps. |
| `--val-every N` | `200` | Validate every N steps. |
| `--val-batches N` | `16` | Score this many validation batches per pass, from one seeded draw across the val rows of every view. |
| `--report-every N` | `10` | Report the train loss every N steps. |
| `--report JSON` | None | Write the run log here. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--no-wired-limit` | Off | Leave the wired limit where it is. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache at this many GB. |
| `--cpu` | Off | Run on the CPU device. |

### distill eval

`eval` scores the student on held-out text, tasks and chat sets, with or
without the adapter. Its task files are jsonl. `arc_easy.jsonl` and
`hellaswag.jsonl` hold `{id, query, choices, gold}` rows, `gsm8k.jsonl`
holds `{id, question, answer}` rows, and `gsm8k_shots.jsonl` holds the
worked examples shown before each question.

These flags control `gmlx distill eval`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--student GGUF` | Required | Score this student GGUF. |
| `--adapter GGUF` | None | Apply this GGUF adapter. |
| `--md PATH` | Required | Write the Markdown report here. |
| `--json PATH` | Required | Write the JSON report here. |
| `--cache DIR` | None | Check the slices for overlap against this cache's corpus. |
| `--slice NAME=PATH` | None | Score this held-out text slice. Repeat the flag for more slices. |
| `--teacher-bpb JSON` | None | Show these teacher bits per byte beside the student's, from a `{slice: bpb}` map or an earlier eval report. |
| `--tasks-dir DIR` | `.` | Read the four task files from this directory. |
| `--tasks LIST` | None | Run these comma-separated tasks, from `arc_easy`, `hellaswag` and `gsm8k`. |
| `--task-limit N` | All | Score this many items per task. |
| `--gsm8k-max-tokens N` | `384` | Give each GSM8K item this generation budget. |
| `--before` | Off | Also score with the adapter disabled in process. The flag needs `--adapter`. |
| `--chat-slice NAME=PATH` | None | Score this jsonl of `{messages}` conversations on their assistant turns, `student_messages` first. The flag repeats. |
| `--chat-sanity PATH` | None | Score this jsonl of `{id, messages, kind}` prompts, where `kind` is `task` or `refuse`, for template compliance and drift from an earlier report's replies. |
| `--chat-max-tokens N` | `256` | Give each chat sanity reply this token budget. |
| `--chat-refs JSON` | None | Anchor the drift score on the replies of this earlier eval report. `--before` overrides it and anchors on the adapter-off replies. |
| `--chat-max-len N` | `2048` | Score chat and reply rows of up to this many student tokens. A longer row loses turns until it fits, or is dropped. |
| `--chat-per-turn` | Off | Score every assistant turn as its own row. |
| `--reply-slice NAME=PATH` | None | Score this jsonl of conversations on the final reply. The flag repeats. |
| `--reply-think` | Off | Reply slices target the final turn from its reasoning trace onward. |
| `--reply-positions JSON` | None | Restrict the reply slices to this census JSON's `high_delta` maps. A file naming none of their rows, or with a frame other than `--reply-think`'s, is refused. |
| `--kld-cache DIR` | None | Score sparse KL against this same-vocabulary cache, which is refused on another tokenizer or a cached id beyond the student's head. |
| `--kld-rows N` | All | Score this many rows of the KL cache, spread over its length order. |
| `--frame-kwargs JSON` | None | Pass these chat-template kwargs to every render. |
| `--max-len N` | `512` | Measure bits per byte in windows of this many tokens. |
| `--bpb-prefix TEXT` | None | Place this text before every window (`\n`, `\t`, `\r` and `\\` decoded), or `@continue` or `@model` for that frame's template prefix. Another `@` exits 2. |
| `--batch-size N` | `8` | Score this many windows per batch. |
| `--cache-limit-gb F` | `4.0` | Cap the MLX buffer cache at this many GB. |
| `--decontam-threshold F` | `0.01` | Void a slice's gate when more than this fraction of its windows is found in the corpus. |
| `--hf-source ID` | None | Replace the config synthesized from the GGUF with this Hugging Face repo's config.json. The tokenizer always comes from the GGUF. |
| `--cpu` | Off | Run on the CPU device. |

### distill census

`census` pairs the reply rows of a cache made without a context with the
same rows in one or more caches made with one, and runs on the CPU. It
reports how much more likely the context makes each token the teacher
wrote, and the distance between the two stored top-k distributions with
everything outside the top-k pooled. With several contexts it also reports
the part no single adapter can learn. Every `--with` cache then decides
which rows pair and which positions count, while the effect, the histogram
and the positions map come from the first `--with` cache.

The action exits 2 when a cache has no manifest
or one it cannot read, or when no rows pair. It also exits 2 when a
`--with` cache was made with another teacher, tokenizer, top-k or head
width than `--without`, and when a reply-think cache records no
`content_start`, which older caches lack. A `--corpus`
that names no file, or holds a line that is not a JSON object, exits 2 as
well.

These flags control `gmlx distill census`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--without DIR` | Required | Read the same replies without the context from this cache. |
| `--with DIR` | Required | Read a cache made with a context. Repeat the flag for more contexts. |
| `--out JSON` | Required | Write the census JSON here. |
| `--md PATH` | None | Write a Markdown summary here. |
| `--corpus JSONL` | None | Key `high_delta` by row id, using the corpus jsonl the caches were made from. |
| `--delta-threshold F` | `1.0` | A position is high-delta when the context adds this many nats at the token the teacher wrote. |
| `--pair-by MODE` | `line` | Pair rows across caches by corpus `line` or by the full `doc` id. |
| `--max-rows N` | All | Measure this many paired rows. |
