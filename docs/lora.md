# LoRA on a quantized GGUF

This guide is for anyone who wants to fine-tune a GGUF model without
converting it, or serve one base model under several adapters. The first half
trains an adapter with `gmlx train`; the second serves a base with many
adapters as separate model ids.

- [Why train on the quant](#why-train-on-the-quant)
- [Train an adapter](#train-an-adapter)
- [Use the adapter](#use-the-adapter)
- [Serving one base with many adapters](#serving-one-base-with-many-adapters)
- [Adapter format and interop](#adapter-format-and-interop)
- [Limitations](#limitations)

## Why train on the quant

`gmlx train` fine-tunes a K-quant GGUF base as it is and writes the adapter
as a small GGUF file. `run`, `chat` and `serve` attach it live at load with
`--adapter`, so one base serves any number of adapted variants, each an exact
delta on the unmodified quantized weights.

Two properties make this worth using over the usual convert, fine-tune,
requantize cycle. The frozen base stays in its K-quant codec during training,
with the adapter's gradient flowing through the quantized matmul, so there is
no float copy of the base and no optimizer state for it: you can fine-tune a
model you could not hold in fp16. And at inference the base bytes are never
modified, so the output is the base with its existing quantization error plus
the exact adapter delta in full precision. Merging would force a
requantization of the adapted weights.

If you have the full-precision model and the memory to spare, fine-tune that
and quantize afterward. Training on the quant is for when the quant is all
you can fit.

## Train an adapter

The walkthrough teaches Qwen3-0.6B to talk like a pirate. Any pure K-quant or
legacy-codec GGUF works as a base:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --to .
```

`--data` takes a directory of `train.jsonl` and `valid.jsonl`, or a Hugging
Face dataset id, in any format mlx-lm's LoRA trainer accepts: chat records
(`{"messages": [...]}`), prompt and completion pairs, or plain text. Chat
records suit an instruct base, since the trainer applies the base's own chat
template. The example dataset,
[GPT007/pirate_speak](https://huggingface.co/datasets/GPT007/pirate_speak),
ships 100 turns as Llama-3-formatted text, so a short script re-emits them as
chat records. It needs the `datasets` package, which gmlx does not install:

```python
# prep_pirate.py
import json, re
from pathlib import Path
from datasets import load_dataset

turn = re.compile(r"user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>.*?"
                  r"assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL)
records = [{"messages": [{"role": "user", "content": m.group(1).strip()},
                         {"role": "assistant", "content": m.group(2).strip()}]}
           for row in load_dataset("GPT007/pirate_speak", split="train")
           if (m := turn.search(row["text"]))]
out = Path("pirate-data"); out.mkdir(exist_ok=True)
split = max(1, len(records) // 10)
(out / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[:split]))
(out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[split:]))
```

Then train. The adapter targets the attention and MLP projections of the top
`--num-layers` layers:

```sh
pip install datasets && python prep_pirate.py
gmlx train Qwen3-0.6B-Q8_0.gguf --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 --adapter-out pirate-lora.gguf
```

Train loss should fall steadily. With only 90 examples, stop around 150
iterations; pushing further overfits, validation loss climbs and greedy
decoding can fall into loops. `--num-layers` and `--rank` trade capacity for
memory, and the defaults of 8 layers at rank 8 are a sensible start. The
walkthrough targets a dense base; on a MoE base the default adaptation keys
are untested. The flag table is under [gmlx train](cli.md#gmlx-train).

## Use the adapter

```sh
gmlx run Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf --prompt "What's the weather like today?"
gmlx run Qwen3-0.6B-Q8_0.gguf --prompt "What's the weather like today?"    # the untouched base
gmlx serve Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf
```

Qwen3 is a thinking model, so `run` emits a `<think>` block first. The pirate
data has no thinking, so the adapted model thinks briefly and gets straight
to the arrr.

The single-model `serve` form registers the adapted model under the
file-derived id and the bare base as `<id>-base` on the same loaded model, so
both are addressable without a config.

## Serving one base with many adapters

One quantized base can serve any number of adapted variants at once. The base
weights load a single time, each adapter loads into its own slot on that
resident model, and every request applies only the adapter of the model id
it addressed. Requests to the base and to any adapted id batch together into
one decode step. An adapter adds about 1% to 2% to decode and prefill cost,
and switching between ids costs nothing because nothing is swapped.

Model ids whose entries name the same `path` and differ only in `adapter:`
share one resident model:

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

| Id | Weights | Adapter slot |
|----|---------|--------------|
| `qwen3-0.6b` | the one loaded base | none |
| `qwen3-0.6b-pirate` | the same base | `pirate-lora.gguf` at scale 1.0 for its rows |
| `qwen3-0.6b-formal` | the same base | `formal-lora.gguf` at scale 1.0 for its rows |

Everything that changes how the model is loaded must agree across the group:
same `path`, same `context_length`, same `speculative`, and so on. An id that
differs in more than `adapter:` becomes its own entry with its own copy of
the weights, so keep the group's other keys identical or inherit them from a
profile. The footprint shows as a single entry under `resident_models` on
`GET /v1/metrics`, and `curl localhost:8080/v1/models` lists all three ids.

There is nothing adapter-specific in the API. Each request names an id, and
the server turns that id's adapter on and every other slot off for the rows
of that request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-pirate",
  "messages": [{"role": "user", "content": "Summarize RAID levels."}]
}'
```

Concurrent requests to different ids of the group do not queue behind each
other, and an adapted request's output equals what it would produce running
alone, whatever else is in the batch.

In the chat client, `gmlx chat --server qwen3-0.6b-pirate` connects to a
served id, and `/model <id>` inside the session switches the id the next turn
goes to while keeping the transcript. Because the ids share one resident
model the switch is instant, so you can ask on the base, switch, and have the
adapted model answer the follow-up with the full context. `/model` alone
lists the served ids, and Tab completes them. The client is described in
[chat.md](chat.md).

Adapters interact with two other features:

- Prompt cache. Adapted and bare rows never share a cache entry, since the
  key is salted per adapter set, at the cost of one cached copy per id that
  shares a prefix.
- Speculative decoding. `speculative: true` combines with adapters. Set it on
  every id of the group, since a mismatch would split the entry. When
  concurrent requests exceed the width cap the batch decodes plain until it
  drains, and adapters behave identically on both sides of that switch.

The sorted adapter set is part of what identifies the loaded model. Adding an
id with a new adapter and reloading the config builds a new entry while the
old one ages out, so plan for both footprints being briefly resident, or
restart instead of reloading when the base is large.

Serving base and adapters together at the low cost above needs an mlx-kquant
build with the in-op LoRA epilogue; older builds fall back to plain-op deltas
with the same results at a somewhat higher cost.

## Adapter format and interop

The adapter file is the llama.cpp GGUF LoRA format, what
`convert_lora_to_gguf.py` emits from a PEFT directory: `general.type` of
`adapter`, `adapter.lora.alpha` in the metadata, and per-target `lora_a` and
`lora_b` tensor pairs keyed to base tensor names, with PEFT scaling. So
adapters trained with `gmlx train` load in llama.cpp with `--lora`, and any
PEFT LoRA converted with that script loads here, as do existing community
GGUF adapters built for llama.cpp.

## Limitations

- LoRA only. DoRA on a K-quant base is not supported.
- Targets are the dense linears (q, k, v, o, gate, up, down) and MoE expert
  down-projection stacks. Gate and up expert stacks, embeddings, and expert
  targets on bases the runtime runs under a fused MoE block (gemma, gpt-oss)
  error at load rather than being skipped.
- Text path only. `--adapter` does not combine with `--mmproj`.
- Each model id's adapter is fixed at load. Switching adapters means
  addressing a different id, not a per-request parameter.
- The adapter must match the base architecture, which is checked at load.
  Matching the exact base fine-tune is your responsibility.
