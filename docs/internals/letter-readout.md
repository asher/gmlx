# Letter readout

The letter readout answers `POST /v1/systemone` on any model other than
DiffusionGemma, with the prompt and calibration of OpenJev's helper. A
decision reads the state once and runs each question's prompt as a short
tail after it, in fixed forwards that the server's batch loop interleaves
with chat. The request and response contract is in
[Structured decisions](../decisions.md), and the timings are in
[Structured read measurements](structured-read-measurements.md#letter-readout-timings).

- [Origin](#origin)
- [One question](#one-question)
- [The shared prefix](#the-shared-prefix)
- [Letter scores](#letter-scores)
- [Running on the server](#running-on-the-server)
- [Kept prefixes](#kept-prefixes)
- [Parity with the helper](#parity-with-the-helper)

## Origin

`gmlx/systemone/letters.py` is a port of `helper/shim.py` from
`huggingface.co/openjev/openjev` at revision `0b6bb6e`, with the settings
that the repository's `serve/SERVE.md` measured. The helper is a proxy that
sends each question to a vLLM server and reads the letter log-probabilities
back. gmlx runs the same protocol in the server process, and
`licenses/openjev-LICENSE` names the ported files.

`letters.py` is pure Python. It holds the request rules, the prompt text of
every pass and the answers, and each question is a generator that yields
the passes it needs and receives their letter scores. `ar_reader.py` runs
the passes on the model, `prefixes.py` keeps state prefixes in the prompt
cache, and `backends.py` parses a body for both readouts, since the route
learns the model's kind only after it reads the body.

## One question

Each question becomes the only user message of a chat prompt, rendered
with the model's template, thinking off and the generation prompt on:

```text
State:
{state}

Question: {instructions}
Options:
[A] {name}: {description}
[B] {name}: {description}

Answer with the letter of the best option only.
```

An object state is its JSON with the characters kept, and a list or number
is its Python string. Instructions that are not a string also take their
Python string, as the helper's `pyrepr` style does. The answer is the score
of each listed letter at the first reply position, divided by 0.85 and
normalized over those letters. Each letter must be a single token of its
own, which the route checks once per model.

A yes or no question lists `yes` and `no`, described by `criteria.true` and
`criteria.false` or by "The statement is true." and "The statement is
false.". Its answer is `sigmoid(logit(p_yes) / 1.829074)`, the helper's
calibration. A score question lists its levels as `0` to `n-1`, its
instructions gain " Rate along the ordered levels below (lowest first).",
and the score is the expected level.

The letters run from `A` to `Z` and then `a` to `z`, so one prompt holds 52
options. A question with more is read in near-equal chunks of at most 52,
then once more over the chunk winners, and the two reads compose as
`p(i) = p_final(chunk of i) * p_chunk(i) / p_chunk(winner of that chunk)`.
With `samples: N`, each chunk is read in N orders, shuffled by
`random.Random(j)` for order `j`, and the probabilities are averaged by
option.

## The shared prefix

Every pass of a request starts with the chat header and the state, so the
reader prefills that part once. The split is the common start of two token
lists, the prompt that ends at the state and the prompt of a pass. Tokens
that merge across the end of the state stay in the tails, and every pass
continues the state with the same text, so any pass gives the same split.

Every forward carries at most `FORWARD_TOKENS` tokens, 128, which never
depends on load. The prefix runs in chunks of that size from position 0,
and each group of tails fills at most one forward. A size that followed
load would change the chunks of the recurrent state and the kernel route,
so the same request could give different numbers. The measurements page
gives the choice between 64, 128 and 256.

Tails run in groups through views of the prefix cache that broadcast it to
every row of the forward. An attention view returns the prefix keys and
values joined to the rows' own and stores nothing. A recurrent view hands
each row the prefix state and drops the writes. Right padding needs no
mask, since no later position changes an earlier one. `plan_buckets` sorts
the tails by length and groups them so that padded tokens plus a fixed
cost per forward is lowest, within the forward size and a 512 MB bound on
the copy that the views make.

A tail longer than one forward, or a cache with layers the views do not
cover, such as a rotating window, runs alone on a bitwise copy of the
prefix. A layer that cannot be copied has its prefix computed again.

## Letter scores

The reader takes the hidden state at each tail's last token and multiplies
it by the head's 52 letter rows in float32. The rows are dequantized once
per model. A bfloat16 head output can round a score by about 0.06, which
the division by 0.85 would carry into the answer.

A self-check picks this path once per model, and again after an adapter
replaces the head. It runs a short probe through the whole model and
through the trunk and head separately, and it takes the rows path only when
the two logits are equal and the float32 rows reproduce the head's letter
scores. Any other model, such as one that scales its logits, reads the
letter columns of the full logits. `diagnostics.path` reports `rows` or
`logits`.

## Running on the server

An autoregressive model's engine thread runs a batch loop that calls
`ResponseGenerator._collect_pending_requests` between batch steps. A
wrapper in `gmlx/serve/engine_jobs.py` takes decision jobs out of the
request queue there, and out of the queue directly, so a full batch cannot
hold a job back. While chat rows are active, it runs one forward of the
first job per call and rotates the jobs. With no chat rows, the loop keeps
calling back at once, so a decision runs at full speed.

A job step leaves the chat rows as it found them. It never drains
cancellations, never touches the tokenizer's stop criteria and never
reseeds the random state. It puts back the rope position state of a
multimodal model, sets the prefill command-buffer phase and restores the
previous one, and holds the wired limit. With per-row adapters, it
publishes the request's adapter scales around each forward.

The deadline, `token_queue_timeout_s`, starts when the job leaves the
queue, and the job's stop event is checked at each forward. A client that
disconnects sets it too. An engine cancel for the job's id never reaches
the job, since the batch loop consumes the id and finds no row, so the
route cancels only through the stop event. A diffusion job receives engine
cancels, and `tests/serve/test_engine_jobs.py` pins both behaviors.

Before queueing, the route tokenizes every pass that the decision can send
and checks the longest against the context and memory budgets, with room
for one forward past it. A composed question's winners pass is bounded by
the longest option of each chunk.

## Kept prefixes

The route keeps each decision's prefix in the server's prompt cache, the
APC manager, so the next decision on the same state reads only its tails.
A model with recurrent layers and full attention, such as OpenJev, keeps it
in the checkpoint tier as a record of kind `decision`, memory only. A
model whose layers are all plain attention keeps it in the exact tier. A
lookup asks for exactly the prefix length, so a shorter record never
serves.

The records carry their own salt, made of the reader version, the forward
size and the request's adapter scales. A chat request never adopts one, and a
decision never adopts a chat record, which was prefilled in other chunks.
The reader's caches are never quantized, so a kept prefix equals a
computed one bit for bit.

Decision records keep to themselves in the checkpoint tier. One is never
promoted to anchor, a decision insert strips only decision records on its
chain, and no other insert strips a decision record. The manager keeps at
most eight of them, so a stream of new states cannot push the chat records
out, and eviction ranks them with the other records that are not anchors.
The exact tier shares its count and byte bounds with chat entries.

A cache stack that neither tier takes reports `"tier": "unsupported"`, and
a server with the prompt cache off, like the offline command line, reports
`"off"`. `POST /v1/prewarm` runs the prefix step alone, as the helper's
route of the same name does.

## Parity with the helper

`tests/systemone/test_systemone_letters.py` runs the vendored helper,
`tests/systemone/_openjev_helper_ref.py`, and the letter decision on the
same scripted letter scores. Both must send the same prompts and return
the same answers for each question type, object and list states, string,
object and list instructions, 52, 60 and 120 options and several orders.

The intended differences are few. The helper rounds its answers to four
decimals and gmlx does not. `confidence` is the official Jev value, as on
DiffusionGemma. The route refuses a state that holds an image, which the
helper would send to a vision model. `ask` and `ask_if` apply, and a
question with `ask_if` is read in a later stage than the questions it
names, although its prompt does not carry their answers.

`tests/systemone/test_systemone_ar_reader.py` checks the reader on a
small Qwen3.5-shaped model. Batched tails must give the scores of a full
forward over the whole prompt, a kept prefix must equal a computed one
through both cache tiers, and a row's scores must not change with the
other rows of its forward.
