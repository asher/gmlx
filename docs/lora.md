# LoRA on GGUF: train an adapter, apply it live

`gmlx` does LoRA GGUF in, GGUF out: `gmlx train` finetunes a K-quant
GGUF base as-is and writes the adapter as a small GGUF file. `run --adapter` /
`chat --adapter` / `serve --adapter` attach it live at load, so one base
serves any number of adapted variants, each an exact delta on the unmodified
quantized weights.

Two properties make this worth using over the usual convert-finetune-requantize cycle:

- Memory: the frozen base stays in its K-quant codec during training. The
  adapter's gradient flows through the quantized matmul (the kquant op defines a
  `vjp`), so there is no float copy of the base and no optimizer state for it. You
  can finetune a model you couldn't hold in fp16.
- Quality: at inference the base wire bytes are never modified. Output = base
  (with its existing quant error) + the exact adapter delta in full precision.
  Merging would force a requantization of the adapted weights; live apply doesn't.

If you do have the full-precision model and the memory to spare, finetune that
and quantize afterward. The adapter then builds on a base without quantization
error. Training on the quant is for when the quant is all you can fit.

## Walkthrough: teach Qwen3-0.6B to talk like a pirate

### 1. Get a base

Any pure K-quant / legacy-codec GGUF works. A small one to follow along with:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --to .
```

### 2. Build the training data

`--data` takes a directory of `train.jsonl` / `valid.jsonl` (or an HF dataset id) in
any format mlx-lm's LoRA trainer accepts: chat (`{"messages": [...]}`),
prompt/completion, or plain text. Chat records are best for an instruct base: the
trainer applies the base's own chat template.

This prep script uses the tiny
[`GPT007/pirate_speak`](https://huggingface.co/datasets/GPT007/pirate_speak)
dataset: 100 chat turns shipped as Llama-3-formatted text, so it pulls out the
user/assistant turns and re-emits them as chat records. It needs the `datasets`
package, which gmlx doesn't install. Passing `--data <HF dataset id>` to
`train` needs it too:

```sh
pip install datasets
```

```python
# prep_pirate.py
import json, re
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("GPT007/pirate_speak", split="train")
turn = re.compile(
    r"user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>.*?"
    r"assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL,
)
records = []
for row in ds:
    m = turn.search(row["text"])
    if m:
        records.append({"messages": [
            {"role": "user", "content": m.group(1).strip()},
            {"role": "assistant", "content": m.group(2).strip()},
        ]})

out = Path("pirate-data"); out.mkdir(exist_ok=True)
split = max(1, len(records) // 10)   # 10% validation
(out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[:split]))
(out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[split:]))
print(f"wrote {len(records) - split} train / {split} valid -> {out}/")
```

```sh
python prep_pirate.py
# wrote 90 train / 10 valid -> pirate-data/
```

### 3. Train the adapter

gmlx runs the training on mlx-lm's LoRA tuner. The adapter targets the attention + MLP
projections of the top `--num-layers` layers:

```sh
gmlx train Qwen3-0.6B-Q8_0.gguf \
    --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 \
    --adapter-out pirate-lora.gguf
```

Train loss should fall steadily. With only 90 examples, stop around 150 iters.
Pushing further overfits: validation loss climbs and greedy decoding can fall into
"me hearty, me hearty..." loops. Turn `--iters` down or add data to taste.
`--num-layers` and `--rank` trade capacity for memory: more adapted layers and a
higher rank fit more behavior but cost more training memory. The defaults (8 layers,
rank 8) are a sensible start. Note the walkthrough targets dense bases. On a MoE
base the default adaptation keys are untested. Full flag
table: the [train section of docs/cli.md](cli.md#gmlx-train).

### 4. Generate with the adapter (no merge)

```sh
# adapted
gmlx run Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf \
    --prompt "What's the weather like today?"

# drop the flag for the plain base: the base file was never touched
gmlx run Qwen3-0.6B-Q8_0.gguf --prompt "What's the weather like today?"
```

Qwen3 is a thinking model, so `run` emits a `<think>` block first. The pirate data
has no thinking, so the adapted model thinks "empty" and gets straight to the arrr.

### 5. Serve it

```sh
gmlx serve Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf --port 8080
```

In config mode, `adapter:` is a per-model key. Model ids on the same GGUF
that differ only in `adapter:` share one resident entry: the base weights load
once, every adapter of the group loads into its own slot, and each request
turns its id's adapter on (scale 1.0) and the others off (0.0) for its rows.
`base`, `base+adapterA` and `base+adapterB` are three ids that serve from one
loaded model and batch together, so switching adapters between requests costs
nothing and a mixed batch of base and adapted rows is one decode step:

```yaml
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

The single-model form (`serve model.gguf --adapter x.gguf`) registers the
bare base as `<id>-base` on the same entry, so both are addressable without
a config. The sorted adapter set is what enters the load signature: a reload
that adds an id with a new adapter builds a new entry and the old one ages
out. Adapted and bare rows never share a prefix cache entry (the APC key is
salted per adapter set), and an adapted request's outputs equal its solo run
whatever else is in the batch. Serving base and adapters together needs an
`mlx-kquant` build with the in-op LoRA epilogue (`HAS_LORA_EPILOGUE`);
older builds fall back to plain-op deltas with the same results.

Serving one base with several adapters (config grouping, the chat client's
mid-conversation `/model` switch, batching and speculative-decoding
behavior) has its own guide: [the serving section below](#serving-one-base-with-many-adapters).

The whole loop above is automated as an end-to-end test,
`python tests/e2e/run_lora_e2e.py`: prep, then train, then serve base and
base+adapter, then assert the pirate voice took.

## Adapter format & interop

The adapter file is the llama.cpp GGUF LoRA format, what
`convert_lora_to_gguf.py` emits from a PEFT directory: `general.type = "adapter"`,
`adapter.lora.alpha` in the KV, and per-target `<base>.weight.lora_a` /
`<base>.weight.lora_b` tensor pairs keyed to base GGUF tensor names, with PEFT
scaling semantics (`delta = (alpha / rank) * B * A`). That buys interop both ways:

- adapters trained with `gmlx train` load in llama.cpp (`--lora`), and
- any PEFT LoRA, via `convert_lora_to_gguf.py`, loads here, as do existing
  community GGUF adapters built for llama.cpp.

## Limitations

- LoRA only: DoRA on a K-quant base is not supported. mlx-lm's DoRA dispatch
  doesn't route through the quantized path.
- Dense linears (q/k/v/o, gate/up/down) and MoE expert `down_proj` stacks.
  Gate/up expert stacks and embeddings error loudly rather than being
  silently skipped, as do expert targets on bases the runtime places under
  a fused MoE block (gemma, gpt-oss).
- Text path only: `--adapter` doesn't combine with `--mmproj` (VLM).
  `--speculative` (native-head MTP) works; see
  [the serving section below](#serving-one-base-with-many-adapters).
- Each model id's adapter is fixed at load: switching adapters means
  addressing a different model id (several ids share the one loaded base,
  see [the serving section below](#serving-one-base-with-many-adapters)), not a per-request
  parameter.
- The adapter must match the base architecture (checked at load). Matching the
  exact base finetune is your responsibility.

---

## Serving one base with many adapters

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
single-model quickstart live in [the training walkthrough above](#walkthrough-teach-qwen3-06b-to-talk-like-a-pirate); the adapter file format
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
