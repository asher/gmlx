# LoRA adapters

A LoRA adapter is a small file that changes how a model answers without
changing the model file. `gmlx train` trains one on a GGUF model, and
`run`, `chat` and `serve` apply it at load:

```sh
gmlx train qwen3-0.6b-q8 --data ./pirate-data --adapter-out ~/models/pirate-lora.gguf
gmlx run qwen3-0.6b-q8 --adapter ~/models/pirate-lora.gguf --prompt "What's the weather like today?"
gmlx serve ~/models/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --adapter ~/models/pirate-lora.gguf
```

gmlx trains on the quantized GGUF as it is and keeps no full-precision
copy, so you can fine-tune a model that fits in memory only when quantized.

- [Training data](#training-data)
- [Train an adapter](#train-an-adapter)
- [Use the adapter](#use-the-adapter)
- [Serving one base with many adapters](#serving-one-base-with-many-adapters)
- [Adapter format and interop](#adapter-format-and-interop)
- [Limitations](#limitations)

## Training data

`--data` takes a folder with `train.jsonl`, and optionally `valid.jsonl`.
Each line is one record in one of these formats:

| Format | Record |
|--------|--------|
| Chat | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| Prompt and completion | `{"prompt": "...", "completion": "..."}` |
| Plain text | `{"text": "..."}` |

Use chat records for an instruct model, because the trainer applies the
model's chat template. The loss covers the prompt as well as the reply.
`--data` also takes a Hugging Face dataset id, which needs the `datasets`
package in the gmlx environment.

## Train an adapter

This example teaches Qwen3-0.6B to talk like a pirate. It uses a dense Q8_0
model, which is the tested case, though the trainer accepts the other GGUF
codecs and plain MLX models too. Download the model, which registers it as
`qwen3-0.6b-q8`. `gmlx pull` saves it in your first model folder, which is
`~/models` on this page:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
```

The [GPT007/pirate_speak](https://huggingface.co/datasets/GPT007/pirate_speak)
dataset has 100 conversations as Llama-3-formatted text. This script writes
them as chat records:

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

Then train:

```sh
pip install datasets && python prep_pirate.py
gmlx train qwen3-0.6b-q8 --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 --adapter-out ~/models/pirate-lora.gguf
```

The training loss should fall steadily. With only 90 examples, stop at about
150 iterations. Longer training overfits, and the model starts to repeat
itself.

The adapter covers every linear layer in the top `--num-layers` layers.
`--num-layers` and `--rank` trade what the adapter can learn against
memory, and the defaults of 8 layers at rank 8 are a good start. When a
longer `--max-seq-length` runs out of memory, add `--grad-checkpoint`,
which needs `--dropout 0` and costs time. [gmlx train](cli.md#gmlx-train)
lists the flags.

## Use the adapter

```sh
gmlx run qwen3-0.6b-q8 --adapter ~/models/pirate-lora.gguf --prompt "What's the weather like today?"
gmlx run qwen3-0.6b-q8 --prompt "What's the weather like today?"    # The model without the adapter.
```

Qwen3 is a thinking model, so `run` prints its reasoning first. The adapted
model thinks briefly and then answers like a pirate.

In `gmlx chat`, `/adapter off` and `/adapter on` turn the adapter off and
on for the next turns, and `/adapter 0.5` scales it.

`gmlx serve` with `--adapter` needs no config file. It serves the adapted
model under an id from the model's file name, here `qwen3-0.6b`, and the
model without the adapter as `qwen3-0.6b-base`:

```sh
gmlx serve ~/models/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --adapter ~/models/pirate-lora.gguf
```

## Serving one base with many adapters

A server can offer one model under several adapters at once. Give each
adapter its own id with the same `path`:

```yaml
models:
  qwen3-0.6b-q8:
    path: unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
  qwen3-0.6b-pirate:
    path: unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
    adapter: pirate-lora.gguf
  qwen3-0.6b-formal:
    path: unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
    adapter: formal-lora.gguf
```

The model loads once, and a request applies the adapter of the id it
names. Requests to different ids batch together and do not wait for each
other. To keep one shared copy:

- Keep every other setting the same across the ids, including
  `speculative`. An id that differs in more than `adapter` loads its own
  copy. A shared [profile](config.md#profiles) keeps the settings equal.
- Write a relative adapter path as you would a model path. It resolves in
  [`server.model_dirs`](config.md#servermodel_dirs).
- Adding an adapter and reloading the config builds a new copy, and the old
  one unloads after its idle timeout. For a large model, restart the server
  instead.

To compare adapters in one conversation, run
`gmlx chat --server qwen3-0.6b-pirate` and switch with
[`/model <id>`](chat.md#undo-retry-and-sessions).

## Adapter format and interop

Adapters use the llama.cpp GGUF LoRA format. gmlx loads PEFT adapters
converted with llama.cpp's `convert_lora_to_gguf.py` and the GGUF adapters
published for llama.cpp, and llama.cpp loads gmlx adapters with `--lora`.

## Limitations

- Only LoRA adapters work. DoRA on a quantized model is not supported.
- An adapter can change the dense linear layers and the down projections
  of MoE experts. Loading refuses an adapter that changes the gate or up
  projections of experts, the embeddings, or any expert weight on a model
  that runs a fused MoE block, such as Gemma and gpt-oss.
- Adapters work on text models only, so `--adapter` does not combine with
  `--mmproj`.
- `--grad-checkpoint` is refused on Kimi K3 and DeepSeek-V4.1.
- The loader checks that the adapter is for the model's architecture, but
  not that the model is the one it was trained on.
