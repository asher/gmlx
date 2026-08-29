# Serve one base with many LoRA adapters

One quantized GGUF base can serve any number of LoRA-adapted variants at
once: the base weights load a single time, each adapter loads into its own
slot on that resident model, and every request applies only the adapter of
the model id it addressed. Requests to the base and to any of the adapted
ids batch together into one decode step. An adapter adds about 1-2% to
decode and prefill cost; switching between ids costs nothing because
nothing is swapped.

This guide covers the serving side: the config, the chat client with
mid-conversation id switching, the OpenAI-style API, and how adapters
behave under batching and speculative decoding. Training an adapter and the
single-model quickstart live in [lora.md](lora.md); the adapter file format
is the llama.cpp GGUF LoRA format, so community adapters published for
llama.cpp drop in unchanged.

## Config: ids that share one resident model

Model ids whose entries name the same `path` and differ only in `adapter:`
are grouped onto one resident entry:

```yaml
server:
  port: 8080

models:
  qwen3-0.6b:
    path: Qwen3-0.6B-Q8_0.gguf
  qwen3-0.6b-pirate:
    path: Qwen3-0.6B-Q8_0.gguf
    adapter: pirate-lora.gguf
  qwen3-0.6b-formal:
    path: Qwen3-0.6B-Q8_0.gguf
    adapter: formal-lora.gguf
```

```sh
gmlx serve --config serve.yaml
```

Three ids are served, one model is loaded, both adapters sit in their own
slots. Memory cost is the base once plus the (small) adapters; the
footprint shows as a single entry under `server.resident_models` on
`GET /v1/metrics`.

Everything load-affecting must agree across the group for the ids to share
the entry: same `path`, same `context_length`, same `speculative`, and so
on. An id that differs in more than `adapter:` becomes its own entry with
its own copy of the weights, so keep the group's other keys identical (or
inherit them from a profile).

The no-config form serves one base plus one adapter and registers the bare
base as a sibling id automatically:

```sh
gmlx serve Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf
# serves the adapted model under the file-derived id, the bare base as <id>-base
```

Verify what came up:

```sh
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[].id'
```

## Chat: compare base and adapters in one conversation

`gmlx chat --server` is a plain client for a running server (no tools, no
assistant memory). Point it at any of the served ids:

```sh
gmlx chat --server --port 8080 qwen3-0.6b-pirate
```

Inside the session, `/model` lists the served ids and `/model <id>`
switches the id the next turn is sent to while keeping the transcript. The
server re-reads the conversation under the new id, so you can ask a
question on the base, switch, and have the adapted model answer the
follow-up with full context of both:

```
> /model
[chat] model: qwen3-0.6b-pirate via http://127.0.0.1:8080/v1
  served   qwen3-0.6b  *qwen3-0.6b-pirate  qwen3-0.6b-formal  (/model <id> switches, transcript kept)
> /model qwen3-0.6b
[chat] model qwen3-0.6b-pirate -> qwen3-0.6b (3 turns of transcript kept; the next reply re-reads it under the new id)
```

Tab completes the served ids after `/model `. Because the ids share one
resident model, the switch is instant: the next request simply carries a
different id, and that id's adapter scale is applied to its rows.

## API: the id is the whole interface

There is nothing adapter-specific in the API. Each request names an id;
the server turns that id's adapter on (scale 1.0) and every other slot off
(0.0) for the rows of that request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-pirate",
  "messages": [{"role": "user", "content": "Summarize RAID levels."}]
}'
```

Concurrent requests to different ids of the group do not queue behind each
other: a batch of (base, base+pirate, base+formal) rows is one forward pass
per token, each row under its own adapter scale. An adapted request's
output equals what it would produce running alone, whatever else is in the
batch.

## Interaction with the prompt cache and speculative decoding

- Prompt cache: adapted and bare rows never share a prefix-cache entry.
  The APC key is salted per adapter set, so a prefix computed under one
  adapter is never replayed for another id, at the cost of one cached copy
  per id that shares a prefix.
- Speculative decoding: `speculative: true` (native-head MTP) combines with
  adapters; set it on every id of the group, since it is load-affecting and
  a mismatch would split the entry. The drafter's verify passes run under
  the same per-row adapter scales as plain decode. When concurrent requests
  exceed the model's speculative width cap, the batch converts to plain
  decode until it drains, then speculation resumes; adapters behave
  identically on both sides of that switch.

## Performance and requirements

Measured on the serve path (same binary, bare vs adapted arm, medians over
thermally alternated rounds): decode within 1%, prefill within 2%, no
change in worst-case inter-token hitch. This holds through MoE expert
targets and speculative decoding.

The in-op cost depends on an `mlx-kquant` build with the LoRA epilogue
(`mlx_kquant.HAS_LORA_EPILOGUE`); older builds fall back to plain-op
deltas with identical outputs and a somewhat higher cost. Adapter targets
follow the training-side support matrix (dense linears and MoE expert
`down_proj` stacks; anything else errors loudly at load rather than being
silently skipped); see the limitations list in [lora.md](lora.md#limitations).

## Reloads

The sorted adapter set is part of the entry's load signature. Editing the
config to add an id with a new adapter and reloading builds a new entry
(base loads again) while the old one ages out; plan for both footprints
being briefly resident, or restart instead of reloading when the base is
large.
