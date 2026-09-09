# LoRA on a quantized GGUF

This guide is for fine-tuning a GGUF model without converting it, and for
serving a base model under several adapters. The first half trains an
adapter with `gmlx train`, and the second serves a base with many adapters
as separate model ids.

- [Why train on the quant](#why-train-on-the-quant)
- [Train an adapter](#train-an-adapter)
- [Use the adapter](#use-the-adapter)
- [Serving one base with many adapters](#serving-one-base-with-many-adapters)
- [Adapter format and interop](#adapter-format-and-interop)
- [Limitations](#limitations)

## Why train on the quant

`gmlx train` fine-tunes a K-quant GGUF base as it is and writes the adapter
as a small GGUF file. `run`, `chat` and `serve` attach it live at load with
`--adapter`, and a base serves any number of adapted variants, each an
exact delta on the unmodified quantized weights.

Two properties make this worth using over the usual convert, fine-tune and
requantize cycle. The frozen base stays in its K-quant codec during
training, with the adapter's gradient flowing through the quantized matmul.
There is no float copy of the base and no optimizer state for it, which
lets you fine-tune a model you could not hold in fp16. At inference the base
bytes are never modified. The output is the base with its existing
quantization error plus the exact adapter delta in full precision. Merging
would force a requantization of the adapted weights.

If you have the full-precision model and enough memory, fine-tune that
and quantize afterward. Training on the quant is for when the quant is all
you can fit.

## Train an adapter

The walkthrough teaches Qwen3-0.6B to talk like a pirate. Any pure K-quant or
legacy-codec GGUF works as a base:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --to .
```

`--data` takes a directory of `train.jsonl` and `valid.jsonl`, or a
Hugging Face dataset id, in any format mlx-lm's LoRA trainer accepts. The
formats are chat records of the `{"messages": [...]}` shape, prompt and
completion pairs, or plain text. Chat records suit an instruct base, since
the trainer applies the base's chat template. The example dataset,
[GPT007/pirate_speak](https://huggingface.co/datasets/GPT007/pirate_speak),
contains 100 turns as Llama-3-formatted text. A short script re-emits them
as chat records. It needs the `datasets` package, which gmlx does not
install:

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
iterations. Training longer overfits, validation loss rises and greedy
decoding can repeat. `--num-layers` and `--rank` trade capacity for memory.
The defaults of 8 layers at rank 8 are a reasonable starting point. The
walkthrough targets a dense base. On a MoE base the default adaptation keys
are untested. The flag table is under [gmlx train](cli.md#gmlx-train).

## Use the adapter

```sh
gmlx run Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf --prompt "What's the weather like today?"
gmlx run Qwen3-0.6B-Q8_0.gguf --prompt "What's the weather like today?"    # the untouched base
gmlx serve Qwen3-0.6B-Q8_0.gguf --adapter pirate-lora.gguf
```

Qwen3 is a thinking model, and `run` emits a `<think>` block first. The
pirate data has no thinking. The adapted model thinks briefly and then
answers in pirate speech.

The single-model `serve` form registers the adapted model under the
file-derived id and the bare base as `<id>-base` on the same loaded model.
Both are addressable without a config.

## Serving one base with many adapters

A quantized base can serve any number of adapted variants at once. The
base weights load a single time. Each adapter loads into a slot of its own
on that resident model, and each request applies only the adapter of the
model id it addressed. Requests to the base and to any adapted id batch
together into a single decode step. An adapter adds about 1% to 2% to
decode and prefill cost. Switching between ids costs nothing, because
nothing is swapped.

Model ids whose entries name the same `path` and differ only in `adapter:`
share a resident model:

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
| `qwen3-0.6b` | the loaded base | none |
| `qwen3-0.6b-pirate` | the same base | `pirate-lora.gguf` at scale 1.0 for its rows |
| `qwen3-0.6b-formal` | the same base | `formal-lora.gguf` at scale 1.0 for its rows |

Everything that changes how the model is loaded must agree across the
group, including `path`, `context_length` and `speculative`. An id that
differs in more than `adapter:` becomes a separate entry with a separate
copy of the weights. Keep the group's other keys identical or inherit them
from a profile. The memory use shows as a single entry under
`resident_models` on `GET /v1/metrics`. `curl localhost:8080/v1/models`
lists all three ids.

There is nothing adapter-specific in the API. Each request names an id, and
the server turns that id's adapter on and all other slots off for the rows
of that request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -d '{
  "model": "qwen3-0.6b-pirate",
  "messages": [{"role": "user", "content": "Summarize RAID levels."}]
}'
```

Concurrent requests to different ids of the group do not queue behind each
other. An adapted request's output equals what it would produce running
alone, whatever else is in the batch.

In the chat client, `gmlx chat --server qwen3-0.6b-pirate` connects to a
served id. `/model <id>` inside the session switches the id the next turn
goes to while keeping the transcript. Because the ids share a resident
model the switch is instant. You can ask on the base, switch, and have the
adapted model answer the follow-up with the full context. `/model` alone
lists the served ids, and Tab completes them. [chat.md](chat.md) describes
the client.

Adapters interact with two other features:

- Prompt cache. Adapted and bare rows never share a cache entry, since the
  key includes the adapter set. The cost is a cached copy for each id that
  shares a prefix.
- Speculative decoding. `speculative: true` combines with adapters. Set it
  on each id of the group, since a mismatch would split the entry. When
  concurrent requests exceed the width cap the batch decodes plain until it
  drains. Adapters behave identically on both sides of that switch.

The sorted adapter set is part of what identifies the loaded model. Adding
an id with a new adapter and reloading the config builds a new entry, and
the old one is evicted after its idle time. Plan for both copies being
briefly resident, or restart instead of reloading when the base is large.

Serving base and adapters together at that low cost needs an mlx-kquant
build with the in-op LoRA epilogue. Older builds fall back to plain-op
deltas, with the same results at a somewhat higher cost.

## Adapter format and interop

The adapter file is the llama.cpp GGUF LoRA format, which
`convert_lora_to_gguf.py` emits from a PEFT directory. It has
`general.type` of `adapter`, `adapter.lora.alpha` in the metadata, and
`lora_a` and `lora_b` tensor pairs for each target, keyed to base tensor
names, with PEFT scaling. Adapters trained with `gmlx train` therefore
load in llama.cpp with `--lora`. Any PEFT LoRA converted with that script
loads here, as do existing community GGUF adapters built for llama.cpp.

## Limitations

- LoRA only. DoRA on a K-quant base is not supported.
- Targets are the dense linears q, k, v, o, gate, up and down, plus MoE
  expert down-projection stacks. Gate and up expert stacks, embeddings, and
  expert targets on bases the runtime runs under a fused MoE block, such as
  gemma and gpt-oss, error at load instead of being skipped.
- Text path only. `--adapter` does not combine with `--mmproj`.
- Each model id's adapter is fixed at load. Switching adapters means
  addressing a different id, not a request parameter.
- The adapter must match the base architecture, which is checked at load.
  Matching the exact base fine-tune is your responsibility.
