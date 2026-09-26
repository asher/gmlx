# Structured reads

How `POST /v1/systemone` turns a question set into answer distributions on
a DiffusionGemma model, for contributors changing the route, the decision
logic or the read engine. The request and response contract is in
[decisions.md](../decisions.md), and the timings are in
[structured-read-measurements.md](structured-read-measurements.md).

- [Origin](#origin)
- [One read](#one-read)
- [Samples](#samples)
- [Stages, chunks and thoughts](#stages-chunks-and-thoughts)
- [Running on the server](#running-on-the-server)
- [Parity with the vLLM example](#parity-with-the-vllm-example)

## Origin

The decision logic is a port of the vLLM structured-diffusion example,
`examples/features/structured_diffusion/structured_server.py`, and the
multi-step read loop follows the diffusion branch of vLLM's DiffusionGemma
model. The example is a proxy that drives vLLM over HTTP, while gmlx runs
the same logic in the server process against the model. Each ported file
keeps the Apache-2.0 header that names its source, and
`licenses/vllm-LICENSE` lists them. Code that gmlx adds is in files of its
own.

The code splits into three layers. Apart from `engine.py` and `denoise.py`,
the modules in `gmlx/systemone/` are pure Python. They hold the schema
rules, the answer templates and `decide`, which runs a decision against any
object with `prefill`, `read` and `think`. `extensions.py` adds
`think: "auto"` and the server's request defaults. `engine.py` implements
the three calls on the mlx-vlm model, and a read of more than one step runs
the loop in `denoise.py`. The route is in `gmlx/serve/patches/systemone.py`,
and `run_on_engine` in `gmlx/serve/engine_jobs.py` runs a decision on the
model's engine thread.

## One read

A read answers every question in a group with one denoise step. The prompt
is the chat template with the system text, which lists the questions and
their labels, and the state as the user message. The prompt is prefilled
once, and every read of the group reuses that cache.

The canvas holds the answer template, such as `urgent: yes`, then the turn
close token and padding to the read's width. The width is the template
length rounded up to a multiple of 16, capped at `server.systemone.canvas`.
At each label position, called a slot, the template's label is replaced by
a random token id. The decoder runs over the canvas once, and the logits at
each slot, restricted to the question's label ids, give the answer.

A slot must be one token that differs between every pair of labels, at the
same position for all of a question's labels. `TemplateResolver` checks
this with the model's tokenizer and refuses the schema otherwise. That is
why labels are `yes` and `no`, letters for choices and digits for scores.

With `constrained` on, the unembedding multiplies the slot rows by the
label rows of the embedding table only, and the log-probabilities are
normalized over the union of the read's label ids. Off, it runs over the
full vocabulary. Both run the matrix product in the activation dtype and
apply the model's own softcap in fp32, as vLLM does. On the unit fixture
the two modes agree after renormalization to within 1e-6.

At one step, then, a read gives each question the same label probabilities
in both modes. The mode changes the entropy over the returned ids, which
decides whether `samples: "auto"` reads more, the `label_mass` and
`argmax_is_label` diagnostics, the loop past one step, and the cost.

A decoder pass reads the prompt K/V through per-layer views instead of a
mutable cache. Each view carries the keys and values, the prompt length
and a marker, and the stock attention code reads them without a patch.
Sliding-window layers get the last `sliding_window - 1` positions, so the
decoder masks come out empty and the read passes none.

With `steps` above 1 the read runs its own denoise loop. The template
positions stay pinned at their seed values, the slots are resampled at the
schedule temperature, and self-conditioning feeds each step's soft
embedding to the next. A sample stops when its argmax canvas has been
stable over the model's stability window with mean entropy under the
confidence threshold, or at the step cap. The answer then comes from the
raw logits of that step.

## Samples

A sample is one read with its own random slot tokens. Sample `k` of a
group uses seed `seed + 7919 k`, and the slot tokens come from Python's
`random.Random` on that seed, so a request with the same seed reproduces
its reads exactly. Samples of one group run as one batch. The prompt views
are broadcast to the batch, and a batch holds at most the model's canvas
length in total canvas positions, so wide canvases split into several
passes.

The default `samples: "auto"` reads once. When the entropy at any slot,
over the returned label set, is above `auto_threshold`, it reads
`auto_max - 1` more samples from base seed `seed + 1`. Each answer is the
mean of the per-sample label probabilities. `decide` also computes the
standard error and the agreement across samples, and the Jev answer shapes
leave both out, as the example's do.

## Stages, chunks and thoughts

Questions with `depends_on` are answered in stages. A later stage's prompt
is the earlier prompt plus the earlier answers as answer lines, so its
slots condition on them. When the new prompt's ids start with the cached
ids, `PromptCache.extend` appends only the difference with a causal update.
Otherwise the prompt is prefilled again, since joining answer lines can
tokenize differently from the lines alone.

A stage whose answer template does not fit the canvas is split into chunks
that are read one after another with consecutive group seeds. The example
reads chunks in parallel. Chunks share one conditioning, so the order does
not change the answers. With `sequential`, each chunk's prompt also carries
the answer lines of the chunks before it, in both implementations.

A request with `think` writes a thought first. The thought runs through
the mlx-vlm denoiser at temperature 1 and the served canvas width, with
the close tag added to the stop set for the call. The global MLX random
state is seeded from the request seed first, so a thought also repeats.
The reads then use the prompt with the thought appended.

`think: "auto"` is a gmlx extension with no counterpart in the example.
`extensions.decide` runs the decision without a thought, and when any
answered question's confidence is below `think_threshold`, runs it again
with a thought of `think_budget` tokens and returns that run. The answers,
`usage`, samples and question diagnostics describe the returned run, while
`timing` adds up both. `diagnostics.think_auto` keeps the first run's
confidences, reads and time, and lists the unsure questions in question
order. The route's admission check prices the prompt with the auto budget.

## Running on the server

mlx-vlm serves a diffusion model from one engine thread, and a decision
runs there as a job. `run_on_engine` queues a request that carries the
job, and a wrapper around the diffusion generate function runs the job in
place of a generation. A chat request to the same model waits behind a
decision, as it waits behind another diffusion generation.

The wait for the engine has no deadline, as for chat. The deadline starts
when the job is dequeued and is `token_queue_timeout_s`. A missed deadline,
or a client that disconnects, sets the job's stop event, and the job
checks it between reads and on every thought token.

Before queueing, the route checks the context and memory budgets against
the largest prompt the decision can reach. That bound starts from the
system text for every question, with the sentence that chunked prompts
add: "A reply may cover only some of the questions; answer every line that
is present." It adds the thought budget and its tags, and one canvas of
answer lines for each chunk of every earlier stage. With `sequential`, the
earlier chunks of the last stage count too.

The route, `run_on_engine` and the engine rely on upstream internals: the
diffusion attention's cache reads, the rotating cache update paths, the
server's diffusion loop and request types, and the prefill-log and
cancellation helpers. Each is fingerprinted in `gmlx/upstream/seams.py`,
so an mlx-vlm upgrade that changes one fails the seams test.

## Parity with the vLLM example

`tests/systemone/test_systemone_proxy_parity.py` runs the vendored example
and the gmlx decision logic on the same scripted reads and compares whole
response bodies for single and chunked stages, `depends_on` stages, skipped
questions, fixed and auto samples, thoughts, multi-step reads and the
indexed format. The intended differences are few. Invalid numbers and
seeds get a 422 instead of a 500, the template cache is bounded, chunks run
one after another, and `ask` returns the asked answers where the example
raises a `KeyError`.

The prompt ids must match too, since every parity claim depends on them.
`scripts/check_dgemma_template.py` renders the decision prompts with the
GGUF's embedded template and compares them with the Hugging Face tokenizer
of `google/diffusiongemma-26B-A4B-it`, the tokenizer the example uses. The
GGUF template is an earlier revision than the one on Hugging Face. The
two render the same ids for plain and thinking prompts, the chunked system
text, the thought tags, answer templates with and without the scaffold,
slot positions and label ids, and a prompt continued by answer lines.

The render passes message content as strings. The mlx-vlm prompt helper
turns content into a list of text parts, and the Gemma 4 template ends
each text part of a system message with a space, which adds one token
before the turn close. Run the script again after a template or tokenizer
change:

```sh
python scripts/check_dgemma_template.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```
