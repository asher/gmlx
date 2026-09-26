# Structured reads

`POST /v1/systemone` turns a question set into answer distributions on a
DiffusionGemma model. A server route, the decision logic and a read engine
do the work. The request and response contract is in
[Structured decisions](../decisions.md), and the timings are in
[Structured read measurements](structured-read-measurements.md).

- [Origin](#origin)
- [One read](#one-read)
- [Samples](#samples)
- [Stages, chunks and thoughts](#stages-chunks-and-thoughts)
- [Running on the server](#running-on-the-server)
- [Parity with the vLLM example](#parity-with-the-vllm-example)

## Origin

The decision logic is a port of the vLLM structured-diffusion example,
`examples/features/structured_diffusion/structured_server.py`, and the
unembedding and multi-step read loop follow the diffusion branch of
vLLM's DiffusionGemma model. The example is a proxy that drives vLLM over
HTTP, while gmlx runs the same logic in the server process. Each ported
file keeps the Apache-2.0 header that names its source, and
`licenses/vllm-LICENSE` lists them.

Three layers make up the code. Apart from `engine.py` and `denoise.py`, the
modules in `gmlx/systemone/` are pure Python. They hold the schema rules,
the answer templates and `decide`, which runs a decision against any
object with `prefill`, `read` and `think`. `extensions.py` adds
`think: "auto"` and the server's request defaults.

`engine.py` implements the three calls on the mlx-vlm model, and `denoise.py` holds
the unembedding and the loop that a read of more than one step runs. The route
is in `gmlx/serve/patches/systemone.py`, and `run_on_engine` in
`gmlx/serve/engine_jobs.py` runs a decision on the model's engine
thread.

## One read

A read answers every question in a group, with one denoise step by
default. The prompt is the chat template with the system text, which
lists the questions and their labels, and the state as the user message.
The prompt is prefilled once, and every read of the group reuses that
cache.

The canvas holds the answer template, such as `urgent: yes`, then the
turn close token and padding. At each label position, called a slot, the
label is replaced by a random token id. At one step the decoder runs over
the canvas once, and the logits at each slot, restricted to the question's
label ids, give the answer.

A slot must be one token that differs between every pair of labels, at the
same position for all of a question's labels. `TemplateResolver` checks
this with the model's tokenizer and refuses the schema otherwise.
That is why labels are `yes` and `no`, letters for choices, and digits for
scores of up to nine levels.

With `constrained` on, the unembedding multiplies the slot rows by the
embedding table's label rows only. With it off, the product runs over
the full vocabulary, and the returned ids also include the argmax token.
At one step both modes give the same label probabilities. The mode changes
the entropy over the returned ids, which decides whether `samples: "auto"`
reads more, and it changes the diagnostics, the loop past one step and the
cost.

A decoder pass reads the prompt K/V through per-layer views instead of a
mutable cache, and the stock attention code reads those views without a
patch.

With `steps` above 1 the read runs its own denoise loop. Every canvas
position other than a slot stays pinned at its seed value, the slots are
resampled at the schedule temperature, and self-conditioning feeds
each step's soft embedding to the next. A sample stops at the step cap, or
earlier when its argmax canvas has been stable over the model's stability
window with mean entropy under the confidence threshold. In constrained
mode that entropy is taken over the label set. Each sample converges on its
own, where vLLM converges per request.

## Samples

A sample is one read with its own random slot tokens. Each sample's
seed derives from the request seed, so a request with the same seed
reproduces its reads exactly. One group's samples run as one batch,
and a wide canvas splits into several passes.

`samples: "auto"` adds reads as
[Samples, steps and thoughts](../decisions.md#samples-steps-and-thoughts)
describes. It checks the first read's entropy over the returned ids, and
one slot above `auto_threshold` is enough to add reads. Each answer is the mean of the per-sample
label probabilities.

## Stages, chunks and thoughts

Questions with `depends_on` are answered in stages. A later stage's
prompt is the earlier prompt plus the earlier answers as answer lines, so
its slots condition on them. When the new prompt's ids start with the
cached ids, `PromptCache.extend` appends only the difference. Otherwise
the prompt is prefilled again, since joined answer lines can tokenize
differently.

A stage whose answer template does not fit the canvas splits into chunks,
read one after another. The example reads chunks in parallel, and since
chunks share one conditioning, the order does not change the answers.
With `sequential`, each chunk's prompt also carries the earlier chunks'
answer lines, in both implementations.

When a request sets `think`, the server writes a thought first, through
the mlx-vlm denoiser, seeded from the request seed so that it repeats. The
reads then use the prompt with the thought appended.

`think: "auto"` is a gmlx extension with no counterpart in the example.
`extensions.decide` implements the two runs that
[Samples, steps and thoughts](../decisions.md#samples-steps-and-thoughts)
describes. `timing` adds up both runs, and
`diagnostics.think_auto` keeps what the first run saw.

## Running on the server

mlx-vlm serves a diffusion model from one engine thread, and a decision
runs there as a job. `run_on_engine` queues a request that carries the
job, and a wrapper around the diffusion generate function runs the job in
place of a generation. A chat request to the same model waits behind a
decision, as it waits behind another diffusion generation.

The wait for the engine has no deadline, as for chat. The deadline starts
when the job leaves the queue, and it is `token_queue_timeout_s`. A missed
deadline, or a client that disconnects, sets the job's stop event,
which the job checks between reads and on every thought token.

Before queueing, the route checks the context and memory budgets against
the largest prompt the decision can reach, including the thought budget
and every earlier stage's answer lines. The route, `run_on_engine`
and the engine rely on upstream internals, each fingerprinted in
`gmlx/upstream/seams.py`.

## Parity with the vLLM example

`tests/systemone/test_systemone_proxy_parity.py` runs the vendored example
and the gmlx decision logic on the same scripted reads and compares whole
response bodies. The intended differences are few. Invalid numbers and
seeds get a 422 instead of a 500, the template cache is bounded, chunks
run one after another, and `ask` returns the asked answers where the
example raises a `KeyError`. The route refuses image states and multipart
bodies, and it does not serve decisions through chat completions.

Every parity claim depends on the prompt ids, so those must match too.
`scripts/check_dgemma_template.py` renders the decision prompts with the
GGUF's embedded template and compares them with the Hugging Face
tokenizer that the example uses. Run it again after a template or
tokenizer change.

```sh
python scripts/check_dgemma_template.py diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```
