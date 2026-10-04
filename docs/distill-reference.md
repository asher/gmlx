# Distillation reference

This page lists the flags and exit codes of every `gmlx distill` action.
The guide is [Distillation](distill.md). Every size flag is in decimal GB,
1e9 bytes.

- [distill gen](#distill-gen)
- [distill filter](#distill-filter)
- [distill cache](#distill-cache)
- [distill align](#distill-align)
- [distill train](#distill-train)
- [distill eval](#distill-eval)
- [distill census](#distill-census)
- [Exit codes](#exit-codes)

## distill gen

Serves a model through `gmlx serve` and writes its replies to a prompt set
as a corpus. A prompt row is `{"id", "messages", "context"}`, with messages
that end on a user turn.

| Name | Default | Meaning |
|------|---------|---------|
| `--out PATH` | Required | Write the corpus jsonl here, with `<out>.gen.json` beside it. A rerun skips the ids already in it. |
| `--prompts PATH` | None | Read prompt rows from this jsonl. |
| `--corpus PATH_OR_ID` | None | Build continuation prompts from this text corpus instead of `--prompts`. |
| `--teacher GGUF` | None | Serve this GGUF: the teacher, or the student for a measurement. `--model` is the same flag. |
| `--base-url URL` | None | Use this running server's `/v1` base instead of serving `--teacher`. |
| `--host HOST` | `127.0.0.1` | Bind the served model to this host. |
| `--port N` | `8093` | Serve the model on this port. |
| `--text-key KEY` | `text` | With `--corpus`, read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | With `--corpus`, read this split of a Hugging Face dataset. |
| `--prefix-chars N` | `1500` | With `--corpus`, quote this many characters of each document in the user turn. |
| `--min-chars N` | `2000` | With `--corpus`, skip documents shorter than this. |
| `--docs N` | All | With `--corpus`, build this many prompts. |
| `--instruction TEXT` | `Continue the following text.` | With `--corpus`, place this user turn before the quoted text. |
| `--chat-template-kwargs JSON` | None | Pass these template variables to every teacher render. The thinking switch has its own flag. |
| `--context FILE` | None | Put this text in the teacher's prompt for every row without its own `context`. The student's prompt goes under `student_messages`. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format. |
| `--thinking` | Off | Turn thinking on and keep the reasoning as `reasoning_content`. Without it, thinking is off. |
| `--thinking-budget N` | None | With `--thinking`, cap the reasoning at N tokens and mark the replies it cut for `filter`. |
| `--tokenizer GGUF_OR_DIR` | `--teacher` | Count the reasoning against the budget with this tokenizer, for `--base-url`. |
| `--serve-arg ARG` | None | Pass this argument to `gmlx serve`. Repeats. Flags that change the prompt or thinking are refused. |
| `--startup-timeout S` | `900` | Wait this many seconds for the served model. |
| `--concurrency N` | `8` | Keep this many requests in flight. |
| `--max-tokens N` | `1024` | Answer budget per request. With `--thinking-budget`, the reasoning has its own budget on top. |
| `--temperature F` | `0.7` | Sampling temperature. |
| `--top-p F` | `0.9` | Keep the most likely tokens whose probabilities add to this. |
| `--top-k N` | The server's | Keep this many candidate tokens. |
| `--min-p F` | The server's | Drop tokens less likely than this share of the best token. |
| `--seed N` | `1` | Base seed. Each request adds its prompt index. |
| `--timeout S` | `1800` | Fail a request after this many seconds. |
| `--report-every N` | `50` | Print a progress line every N replies. |

## distill filter

Drops generated rows a student should not learn from. The drop reasons are
listed in [The filter dropped most rows](distill-troubleshooting.md#the-filter-dropped-most-rows).

| Name | Default | Meaning |
|------|---------|---------|
| `--in PATH` | Required | Read this generated corpus. Repeats, and inputs join in order. |
| `--out PATH` | Required | Write the filtered corpus here, with `<out>.gen.json` beside it. |
| `--report JSON` | None | Write the kept and dropped counts here. |
| `--rejects PATH` | None | Write one `{id, reason}` line per dropped row here, with the checker's word under `detail`. |
| `--min-words N` | `16` | Drop answers with fewer words, reasoning not counted. An ideograph or kana counts as a word. `--min-tokens` is the same flag. |
| `--ngram N` | `8` | N-gram size of the repetition check. |
| `--max-repeat F` | `0.2` | Drop replies whose repeated n-grams exceed this fraction. |
| `--max-trace-repeat F` | `0.5` | Drop replies whose reasoning's repeated n-grams exceed this fraction. |
| `--max-line-repeats N` | `2` | Drop replies with a line repeated more than this many times in a row. |
| `--max-non-ascii F` | Off | Drop replies whose non-ASCII character fraction exceeds this. |
| `--max-reply-tokens N` | Off | Drop replies longer than this many tokens, reasoning included. |
| `--keep-budget-hit` | Off | Keep replies whose reasoning the thinking budget cut. |
| `--verify CMD` | None | Run this shell command as your checker. It reads rows as jsonl on stdin and prints `ok` or a reason word per row. |
| `--context FILE` | None | Put this text on the teacher's side of every kept row, with the prompt as given under `student_messages`. |
| `--context-format FMT` | `{context}\n\n{prompt}` | Combine the context and the last user turn with this format. |

## distill cache

Runs the teacher over a corpus once and stores its likely next tokens at
every position.

| Name | Default | Meaning |
|------|---------|---------|
| `--teacher GGUF` | Required unless `--validate` | Run this teacher GGUF, which may be sharded. |
| `--corpus PATH_OR_ID` | Required unless `--validate` | Read this jsonl file, folder of text files, or Hugging Face dataset id, `id[@config]`. |
| `--out DIR` | Required unless `--validate` | Write the cache to this directory. |
| `--validate DIR` | None | Check an existing cache and exit, without loading a teacher. |
| `--top-k N` | `256` | Keep this many next-token candidates per position. |
| `--max-len N` | `2048` | Cut rows into windows of at most this many teacher tokens. |
| `--max-disk-gb F` | None | Refuse when the size estimate exceeds this. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache during the pass. |
| `--logits-cap-gb F` | `4.0` | Size the head sub-chunk to fit this memory cap. |
| `--floor` | Off | Also store `floor_kld`, the KL against the f16-rounded top-k. |
| `--rows-per-shard N` | `64` | Write this many rows per shard file. |
| `--trunk N` | `512`, or `8192` streaming | Run the trunk in chunks of this many tokens. |
| `--resume` | Off | Continue after the last verified shard. |
| `--max-rows N` | None | Stop after this many rows. |
| `--max-tokens N` | None | Stop after this many teacher tokens. |
| `--limit-docs N` | None | Read at most this many documents. |
| `--text-key KEY` | `text` | Read text from this column of a jsonl or dataset row. |
| `--hf-split NAME` | `train` | Read this split of a Hugging Face dataset. |
| `--source TAG` | `human`, or `synthetic` for a `gen` corpus | Write this source tag on every row. |
| `--frame KIND` | `none` | Where the targets are: `none` plain text, `continue`, `chat` every assistant turn, `reply` the final one, `reply-think` from its reasoning. |
| `--per-turn` | Off | With `chat` or `reply`, write one row per assistant turn. |
| `--student-messages-key KEY` | `student_messages` | The row key that holds the student's own message list. |
| `--frame-instruction TEXT` | `Continue the following text.` | The user turn for the `continue` frame. |
| `--messages-key KEY` | `messages` | Read the conversation the teacher sees from this key. |
| `--close-final-windows` | Off | With `continue`, end a document's last window with the turn-end marker. |
| `--frame-kwargs JSON` | None | Pass these template variables, an object or a file, to every teacher render. |
| `--hf-source ID` | None | Use this Hugging Face repo's config.json instead of the one built from the GGUF. |
| `--no-require-feeder` | Off | Run a streaming teacher without the prefill feeder. |
| `--no-wired-limit` | Off | Leave the wired limit as it is. |
| `--stream-experts` | Off | Stream a MoE teacher's experts even when they fit in memory. |
| `--expert-bytes-gb F` | The streamed bytes | Expert bytes read per forward pass, for the read-traffic report. |
| `--routes` | Off | On a MoE teacher, store the experts chosen at each position, for `eval --kld-cache`. |
| `--hidden` | Off | Also store a sketch of the teacher's final hidden state, for `train --hs`. |
| `--hidden-dim N` | `256` | Width of the hidden sketch. |
| `--hidden-seed N` | `1` | Seed of the sketch matrix. |
| `--cpu` | Off | Run on the CPU. |

## distill align

Maps a cache onto the student's tokenizer and writes the view that `train`
reads.

| Name | Default | Meaning |
|------|---------|---------|
| `--cache DIR` | Required | Read this cache. |
| `--student GGUF_OR_DIR` | Required | Align to this student GGUF, or an MLX checkpoint folder for its tokenizer. |
| `--out DIR` | Required | Write the view here, replacing an earlier view. |
| `--tables DIR` | None | Reuse the tokenizer tables of this earlier view when the pair matches. |
| `--kprime N` | The maximum seen | Keep at most this many student-token groups per boundary. |
| `--materialize` | Off | Also write the batch tensors as view shards. |
| `--max-disk-gb F` | None | Refuse to materialize past this size. |
| `--force` | Off | Keep a view the own-group check would refuse. |
| `--val-fraction F` | `0.02` | Hold back this fraction of rows for validation, whole documents at a time. |
| `--seed N` | `1` | Seed of the validation split. |
| `--w-mid F` | `0.5` | Weight of a shared boundary inside a word. |
| `--gamma F` | `0.001` | Drop chunk-term (ALM) chunks whose teacher boundary mass is under this. |
| `--tau-alm F` | `1.0` | Temperature of the chunk term (ALM). |
| `--T-dk F` | `1.0` | Temperature of the KL term's group softmaxes. |
| `--max-chunk-len N` | `8` | Longest ALM chunk, in tokens on either side. |
| `--frame-kwargs JSON` | None | Pass these template variables to every student render. |
| `--cpu` | Off | Run on the CPU. |

## distill train

Fits a LoRA adapter on the student against one or more views.

| Name | Default | Meaning |
|------|---------|---------|
| `--view DIR` | Required | Train on this view. Repeat to mix views aligned alike over one tokenizer pair. |
| `--student GGUF` | Required | Train this student GGUF, which may be sharded. |
| `--adapter-out PATH` | Required | Write the GGUF adapter here. |
| `--iters N` | Required | Train for this many steps. |
| `--lora-rank N` | `16` | LoRA rank, the adapter's capacity. |
| `--lora-scale F` | `2.0` | LoRA multiplier. |
| `--lora-alpha F` | None | Set the multiplier as alpha over rank instead of `--lora-scale`. |
| `--lora-dropout F` | `0.0` | LoRA dropout, below 1. |
| `--grad-checkpoint` | Off | Recompute activations in the backward pass, to save memory. Not on Kimi K3 or DeepSeek-V4.1. |
| `--lr F` | `1e-4` | Peak learning rate. |
| `--batch-size N` | `8` | Rows per step. |
| `--warmup F` | `0.05` | Warm up over this fraction of the steps, then decay by cosine. `0` starts at the peak. |
| `--weight-decay F` | `0` | AdamW weight decay. |
| `--clip F` | `1.0` | Gradient norm clip. `0` turns it off. |
| `--seed N` | `1` | Seed of the batch order and the LoRA init. |
| `--loss MODE` | `bucketed` | Sparse KL form: `bucketed` with a tail bucket, `paper` without one, or `renorm` over the top-k. |
| `--dk F` | `1` | Weight of the KL term. |
| `--alm F` | `1`, or `0` on the identity path | Weight of the chunk term (ALM). |
| `--ce F` | `0` | Weight of the cross-entropy term. |
| `--T-dk F` | The view's | Override the view's `--T-dk`. |
| `--tau-alm F` | The view's | Override the view's `--tau-alm`. |
| `--gamma F` | The view's | Override the view's `--gamma`. |
| `--chunk N` | `512` | Run the head in chunks of this many positions. |
| `--hs F` | `0` | Weight of the hidden-state term, which needs `cache --hidden`. |
| `--hs-loss MODE` | `cosine` | Compare hidden states by `cosine`, or by `mse` on unit vectors. |
| `--ckpt-dir DIR` | `./ckpt` | Write checkpoints here. A new run refuses a folder that already holds some. |
| `--resume` | Off | Continue from the last checkpoint in `--ckpt-dir`. |
| `--save-every N` | `200` | Save a checkpoint every N steps. |
| `--val-every N` | `200` | Validate every N steps. |
| `--val-batches N` | `16` | Validation batches per pass. |
| `--report-every N` | `10` | Print the training loss every N steps. |
| `--report JSON` | None | Write the run log here. |
| `--hf-source ID` | None | Use this Hugging Face repo's config.json instead of the one built from the GGUF. |
| `--no-wired-limit` | Off | Leave the wired limit as it is. |
| `--cache-limit-gb F` | `8.0` | Cap the MLX buffer cache. |
| `--cpu` | Off | Run on the CPU. |

## distill eval

Scores the student with and without the adapter. The benchmark task files
are local jsonl: `arc_easy.jsonl` and `hellaswag.jsonl` with
`{id, query, choices, gold}` rows, `gsm8k.jsonl` with `{id, question, answer}`
rows, and `gsm8k_shots.jsonl` with the worked examples.

| Name | Default | Meaning |
|------|---------|---------|
| `--student GGUF` | Required | Score this student GGUF. |
| `--adapter GGUF` | None | Apply this adapter. |
| `--md PATH` | Required | Write the Markdown report here. |
| `--json PATH` | Required | Write the JSON report here. |
| `--cache DIR` | None | Mark slices that overlap this cache's corpus. |
| `--slice NAME=PATH` | None | Score this plain-text slice. Repeats. |
| `--teacher-bpb JSON` | None | Show these teacher bits per byte beside the student's, from a `{slice: bpb}` map or an earlier report. |
| `--tasks-dir DIR` | `.` | Read the task files from this folder. |
| `--tasks LIST` | None | Run these comma-separated tasks: `arc_easy`, `hellaswag`, `gsm8k`. |
| `--task-limit N` | All | Score this many items per task. |
| `--gsm8k-max-tokens N` | `384` | Generation budget per GSM8K item. |
| `--before` | Off | Also score with the adapter off, in the same process. Needs `--adapter`. |
| `--chat-slice NAME=PATH` | None | Score this jsonl of conversations on their assistant turns. Repeats. |
| `--chat-sanity PATH` | None | Check template compliance and drift on this jsonl of `{id, messages, kind}` prompts, `kind` being `task` or `refuse`. |
| `--chat-max-tokens N` | `256` | Reply budget for the chat sanity set. |
| `--chat-refs JSON` | None | Measure drift from the replies in this earlier report. `--before` measures it from the adapter-off replies instead. |
| `--chat-max-len N` | `2048` | Score chat and reply rows of up to this many student tokens. |
| `--chat-per-turn` | Off | Score every assistant turn as its own row. |
| `--reply-slice NAME=PATH` | None | Score this jsonl of conversations on the final reply. Repeats. |
| `--reply-think` | Off | Score reply slices from the reasoning onward. |
| `--reply-positions JSON` | None | Score reply slices only at the positions this census JSON names. |
| `--kld-cache DIR` | None | Score the KL against this cache, which must use the student's tokenizer. |
| `--kld-rows N` | All | Score this many rows of the KL cache. |
| `--frame-kwargs JSON` | None | Pass these template variables to every render. |
| `--max-len N` | `512` | Measure bits per byte in windows of this many tokens. |
| `--bpb-prefix TEXT` | None | Place this text before every window, or `@continue` or `@model` for that frame's template prefix. |
| `--batch-size N` | `8` | Windows per batch. |
| `--cache-limit-gb F` | `4.0` | Cap the MLX buffer cache. |
| `--decontam-threshold F` | `0.01` | Void a slice's score when more than this fraction of it is in the corpus. |
| `--hf-source ID` | None | Use this Hugging Face repo's config.json instead of the one built from the GGUF. |
| `--cpu` | Off | Run on the CPU. |

## distill census

Measures how much a document moves the teacher, from a cache of replies
made without it and one or more caches of the same replies made with it.
Runs on the CPU.

| Name | Default | Meaning |
|------|---------|---------|
| `--without DIR` | Required | The cache of the replies without the document. |
| `--with DIR` | Required | A cache of the same replies with a document. Repeat for more documents. |
| `--out JSON` | Required | Write the census JSON here, which `eval --reply-positions` reads. |
| `--md PATH` | None | Write a Markdown summary here. |
| `--corpus JSONL` | None | Key the positions map by row id, from the corpus the caches were made from. |
| `--delta-threshold F` | `1.0` | Count a position as high-delta when the document adds this many nats. |
| `--pair-by MODE` | `line` | Pair rows across caches by corpus `line` or by full `doc` id. |
| `--max-rows N` | All | Measure this many paired rows. |

## Exit codes

Every action exits 0 on success and 2 on a refused input or setting, a
missing flag or input file, or an output path it cannot write. Each checks
its output paths before any model loads.

| Action | Code | Meaning |
|---|---|---|
| `gen` | 1 | Some requests failed. Rerun the same command to retry them. |
| `gen` | 2 | Also: the server failed to start. |
| `filter` | 2 | Also: the `--verify` command failed. |
| `cache` | 2 | Also: a shard could not be written. The verified shards stay for `--resume`. |
| `cache` | 3 | The memory probe missed twice, or a `--routes` recording did not match its rows. |
| `cache` | 4 | The validator failed on what the pass wrote. |
| `cache --validate` | 1 | The cache has a problem. |
| `align` | 3 | The own-group check refused the tokenizer pair. No view is written. |
