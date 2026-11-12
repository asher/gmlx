# Structured decisions reference

This page lists every field of `POST /v1/systemone` and what the response
reports. [Structured decisions](decisions.md) shows how to serve a model and
send a first request.

- [Fields by model](#fields-by-model)
- [Request fields](#request-fields)
- [Questions](#questions)
- [Media on d1](#media-on-d1)
- [Stages and skipped questions](#stages-and-skipped-questions)
- [Samples, steps and thoughts](#samples-steps-and-thoughts)
- [Usage and diagnostics](#usage-and-diagnostics)
- [Repeated states](#repeated-states)
- [Errors and queueing](#errors-and-queueing)
- [How a decision is read](#how-a-decision-is-read)

## Fields by model

DiffusionGemma reads every question of a request in one pass. OpenJev, and
any other text chat model, answers through the
[letter readout](glossary.md#letter-readout): each question is a prompt of
its own, with the options lettered `A)`, `B)` and so on. A model whose file
name or id contains `OpenJev` gets the `[A]` form it was trained on. Both
readouts take the same request and return the same answer shapes. The
letter readout skips the fields it does not use and names them in an
`ignoring unsupported parameter(s)` warning in the server log.

A model whose file ships its own decision prompt, such as Granite Switch,
answers through that prompt instead, as
[How a decision is read](#how-a-decision-is-read) describes. It takes the
same fields as the letter readout.

The d1 models read each question as a prompt of its own, as llama-server
does. They read only the request's `model`, `profile`, `state`,
`questions`, `images` and `files`, and name any other field in the
warning. `adapters` and `lora` get a 400.

| Field | DiffusionGemma | Letter readout | d1 |
|-------|----------------|----------------|----|
| `state` | Required | Required | Required, and can be `null` |
| `instructions` in a question | Optional | Required | Required. On d1-Omni-2B, a spoken clip can take its place |
| `criteria` of a choice or score | 2 to 26 alternatives | 2 or more alternatives | Up to 255 options, or 2 to 10 levels |
| `samples` | Averages reads with different random tokens | Averages reads with different option orders | Ignored |
| `ask_if` | Applied | Applied | Ignored: every question is answered |
| `depends_on`, `alone` | Applied | Ignored | Ignored |
| `steps`, `think`, `think_threshold`, `think_budget`, `auto_max`, `auto_threshold` | Applied | Ignored, and so are the `server.systemone` think defaults | Ignored |
| The request's `instructions`, `chunk_rows`, `chunk_prompt`, `sequential`, `seed` | Applied | Ignored | Ignored |
| `images`, `files` and media parts in a message `state` | `images` gets a 400, `files` is ignored, parts are read as text | As DiffusionGemma | Read, as [Media on d1](#media-on-d1) describes |
| An image under `screenshot` or `image` in an object `state` | Read as text | 400 | Read as text |

## Request fields

| Field | Default | Meaning |
|-------|---------|---------|
| `state` | Required | The text the questions are about. A value that is not a string reaches the model as its JSON text. |
| `questions` | Required | A map of question id to question, as [Questions](#questions) lists. |
| `model` | [`server.systemone.model`](config.md#serversystemonemodel) | A model id or alias. An absent or unknown name falls back to the configured model. |
| `profile` | None | The [profile](config.md#profiles) that `model` resolves with. |
| `ask` | Every question | Only these ids appear in `answers`. The list must include every question they depend on. |
| `instructions` | None | Text added to the system prompt ahead of the questions. |
| `samples` | `"auto"` | How many reads to average. `"auto"` reads once and adds reads when an answer is uncertain. |
| `auto_max` | `4` | The most reads that `"auto"` takes. |
| `auto_threshold` | `0.1` | `"auto"` adds reads when the entropy at a label position is above this value, in nats. |
| `steps` | `1` | Denoise steps for each read, clamped to 1 to 8. |
| `think` | [`server.systemone.think`](config.md#serversystemonethink) | A thought budget of 0 to 4096 tokens, or `"auto"`. The reads see the thought. |
| `think_threshold` | [`server.systemone.think_threshold`](config.md#serversystemonethink_threshold) | Under `"auto"`, a chosen answer's probability below this value runs the decision again with a thought. |
| `think_budget` | [`server.systemone.think_budget`](config.md#serversystemonethink_budget) | The thought budget under `"auto"`, 1 to 4096 tokens. |
| `chunk_rows` | [`server.systemone.canvas`](config.md#serversystemonecanvas), 64 | The most canvas tokens one read's answer template may take, at least 8. A larger stage is split into chunks. |
| `chunk_prompt` | `"own"` | `"own"` gives each chunk a system prompt with only its questions. `"shared"` gives every chunk all of them. |
| `sequential` | `false` | With `true`, the chunks are read in order on one prompt, and each sees the answers before it. |
| `seed` | `42` | The random tokens that each DiffusionGemma read starts from. |
| `adapters` | The entry's strengths | Adapter strengths for the reads, as [Mixing adapters per request](lora.md#mixing-adapters-per-request) describes. |
| `lora` | The entry's strengths | Adapter strengths by index, in the llama-server form. |
| `images`, `files` | None | On d1, data URLs of images, video or audio, as [Media on d1](#media-on-d1) describes. |

`samples` and `auto_max` above
[`server.systemone.max_samples`](config.md#serversystemonemax_samples) are
lowered to it.

## Questions

| Field | Default | Meaning |
|-------|---------|---------|
| `type` | Required | `noul` for yes or no, `choice` for one of several options, `score` for one of several ordered levels. |
| `instructions` | Empty | The question text. Required on the letter readout. |
| `criteria` | Required, except for `noul` | `noul`: `true` and `false` descriptions. `choice`: option names to descriptions. `score`: level names in order. |
| `ask_if` | None | Question ids mapped to lists of their answers. The question is read only when that answer is in the list. |
| `depends_on` | None | On DiffusionGemma, the read sees the answers that the listed questions got first. |
| `alone` | `false` | On DiffusionGemma, with `true`, the question is read on its own. |

Describe each option so that no two overlap, since the probability splits
between options that both fit. The letter readout lists up to 52 options in
one prompt and reads a longer list in chunks, then once more over the chunk
winners. A model's own decision prompt does the same, with as many options
in one prompt as it has labels: `A` to `Z`, then the two-letter labels that
are one token. A question id may not contain a colon or a newline.

An `ask_if` list holds answers of the question named by its key: `"yes"` or
`"no"` for a `noul`, option names for a `choice`, and level names for a
`score`.

On DiffusionGemma, every answer label must be a single token in the answer
template, or the request gets a 422. With more than ten questions, a
numbered id such as `q1` is the safe choice.

## Media on d1

A d1 request takes media in llama-server's forms:

- `images` and `files`, arrays of data URLs. Either one takes an image, a
  video or an audio clip, such as `data:video/mp4;base64,...`.
- Parts of a message-shaped `state`: `image_url`, `video_url` and
  `input_audio`.
- On d1-Omni-2B, a question whose `instructions` is a spoken clip:

```json
{"spoken": {"type": "noul",
            "instructions": {"type": "input_audio",
                             "input_audio": {"data": "UklGRiQ...", "format": "wav"}}}}
```

The model sees the media in that order: `files`, then `images`, then the
message parts. A spoken clip comes after them, in its own question only.

A request holds up to 8 media items. A video's frames are read, not its
sound. Images are sized as in [LFM3 chat](lfm3.md#images-video-and-audio).
Media need the projector file, and audio needs d1-Omni-2B. `videos` gets a
400: put a video in `files`.

## Stages and skipped questions

A question with `depends_on` or `ask_if` is read in a later stage than the
questions it names, and its prompt carries their answers. `ask_if` adds its
keys to `depends_on`. Each stage reads again, so use `depends_on` only for
questions whose answer changes with the earlier one.

When the answer is not in the `ask_if` list, the question is not read. Its
answer is `null`, and `diagnostics.skipped` says why. With
`"ask_if": {"team": ["infra"]}` on the billing ticket of the guide:

```json
"skipped": {"refund": {"because": "team", "was": "billing", "wanted": ["infra"]}}
```

The letter readout ignores `depends_on` and skips questions the same way,
but its prompts do not carry the earlier answers.

## Samples, steps and thoughts

A thought is the costliest setting. The model writes it with its full
denoise loop, while a read without one takes a single pass. Under
`think: "auto"`, the decision first runs without a thought. When any chosen
answer's probability is below `think_threshold`, it runs again with a
thought of `think_budget` tokens, and the answers come from that run.
`diagnostics.think_auto` says whether the thought ran.

The two thresholds point in opposite directions:

- `think_threshold` is a floor on the probability of each chosen answer, so
  a higher value thinks more often. It reads that probability, not
  `confidence`.
- `auto_threshold` is a ceiling on entropy, so a higher value adds reads
  less often.

Without `"auto"`, the server ignores `think_threshold` and `think_budget`
and names them in an `ignoring unsupported parameter(s)` warning.

On the letter readout, `samples: N` reads each question with its options in
N fixed orders and averages the probabilities by option. On OpenJev, four
orders changed no answer in the
[measurements](internals/structured-read-measurements.md#letter-readout-accuracy),
so one order is enough. [Structured read
measurements](internals/structured-read-measurements.md) gives what each
setting costs.

## Usage and diagnostics

`usage.input_tokens` is the longest prompt a read ran on, on DiffusionGemma,
and every question's prompt summed, on the letter readout and d1.
`usage.output_tokens` counts the answer template and any thought on
DiffusionGemma, and is 0 on the letter readout and d1, as OpenJev's own
server and llama-server report.

`diagnostics` describes how the decision ran. Both readouts report the
`stages`, the `skipped` questions and the `timing`. DiffusionGemma adds the
`chunks`, the `thought`, the `samples`, each question's entropies and the
settings it used. The letter readout adds `"readout": "letters"`, the option
`orderings` and the `prefix` that [Repeated states](#repeated-states)
describes. A model's own decision prompt reports the same fields, with
`"readout": "template"`. A d1 model reports `"readout": "d1"`, the
`forwards` it ran and the `timing`.

| Field | Readout | Meaning |
|-------|---------|---------|
| `samples.tops` | DiffusionGemma | One entry per read. Maps each question to its top label as the answer template writes it, that label's probability and the read's entropy. |
| `passes` | Letter readout | How many question prompts ran. |
| `timing.reads` | Both | How many model runs the decision took. On the letter readout, one run can hold several prompts. |
| `prompt_tokens` | Letter readout | Every prompt counted in full. |
| `computed_tokens` | Letter readout | Only the tokens the model ran. The state part counts once, and not at all when the cache had it. |
| `prefix.tokens` | Letter readout | The length of the state part in tokens. |

In `samples.tops`, a score label is its level counted from 1, such as `"2"`
for medium. `probabilities` counts the same levels from 0. A score with ten
or more levels uses the letters `A` onward. When a decision has stages,
chunks or a skipped question, `samples.n`, `samples.tops` and
`samples.policy` are lists with one entry for each chunk of each stage.

## Repeated states

On the letter readout, every question's prompt starts with the state. With
[`server.cache.enabled`](config.md#cacheenabled) on, the server keeps the
state part in its [prompt cache](glossary.md#prompt-cache), and the next
decision on the same state reads only its questions. `diagnostics.prefix`
has `"reused": true` when a decision found a kept state, `"stored": true`
when it kept its own, and the cache `tier`.

`POST /v1/prewarm`, also at `/prewarm`, reads a state before its questions
arrive. It takes the `model`, `state` and `profile` fields of a decision:

```sh
curl localhost:8080/v1/prewarm -d '{"model": "openjev", "state": "Everything is down and we have a demo at noon."}'
```

```json
{"model": "openjev", "ok": true, "prompt_tokens": 17,
 "prefix": {"reused": false, "stored": true, "tier": "ckpt"}}
```

`ok` is true when the state is now kept. The route answers 400 on
DiffusionGemma and d1, which keep no state between requests. A model whose cache
layers the prompt cache cannot store reports `"tier": "unsupported"`, and
with the cache off the tier is `"off"`.

## Errors and queueing

| Status | Cause |
|--------|-------|
| 400 | The body is not a JSON object, carries `images` on a model other than d1, or does not fit the context or memory budget. |
| 400 | On the letter readout, the tokenizer has no single token for each option letter, or an object `state` holds an image. |
| 400 | On d1: media the model cannot read, more than 8 items, `adapters` or `lora`, a question prompt over 65,536 tokens, or more memory than is free. |
| 400 | `decision_model_only`: a route other than this one names a d1 model. |
| 400 | `unknown_profile`: `profile` names no profile. `no_model_specified`: `model` is absent and there is no fallback. |
| 401 | The server has an API key, and the request does not present it. |
| 404 | `model_not_found`: `model` names nothing and there is no fallback. `model_file_missing`: the model file is gone. |
| 422 | `validation_error`: a field fails validation, such as a missing `state` or a value out of range. |
| 503 | The queue is full or the model load is deferred, as [Limits and back-pressure](api.md#limits-and-back-pressure) describes. |
| 503 | `decision_busy`: another d1 decision held the server past `server.token_queue_timeout_s`. The response carries `Retry-After`. |
| 504 | `timeout`: the decision ran past [`server.token_queue_timeout_s`](config.md#servertoken_queue_timeout_s) after it left the queue. |
| 500 | `server_error`: the engine failed. |

On DiffusionGemma, a decision holds the model from its first read to its
last, so a chat request to the same model waits behind it. On the letter
readout, chat on the same model keeps streaming during a decision, more
slowly. A client that disconnects cancels its decision.

## How a decision is read

On DiffusionGemma, the questions and their answers become the system
prompt, and the state becomes the user message. The
[canvas](glossary.md#canvas) holds an answer template with a random token
at each answer position, and one denoise step gives the distribution over
each question's labels. [Structured reads](internals/structured-reads.md)
describes the mechanism.

On any other model, each question becomes a user message that holds the
state, the question and its options under the letters `A`, `B` and so on.
The model's probability for each letter at the start of its reply is the
answer. [Letter readout](internals/letter-readout.md) describes the
mechanism.

A model can ship its own decision prompt as the chat template
`tokenizer.chat_template.systemone`. Granite Switch does: its prompt puts
the state in a user turn of its own, which runs on the base model, and
opens the question turn with the `decision` adapter. That state turn is the
part [Repeated states](#repeated-states) keeps. The answer is the model's
probability for each option label at the start of its reply.

A d1 model renders each question with the decision prompt in its file: the
state, then the question and its labels. Its full attention layers read the
whole prompt in both directions, so the state is read with the question in
view. The questions of one request run side by side in one pass when memory
allows, and each answers as it would alone.
