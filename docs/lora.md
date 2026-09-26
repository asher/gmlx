# LoRA adapters

A LoRA adapter is a small file that changes how a model answers without
changing the model file. `gmlx train` trains one on a GGUF model, `run`,
`chat` and `serve` apply it at load, and one server can offer a model
under several adapters at once.

- [Training on the quantized model](#training-on-the-quantized-model)
- [Train an adapter](#train-an-adapter)
- [Use the adapter](#use-the-adapter)
- [Serving one base with many adapters](#serving-one-base-with-many-adapters)
- [Adapter format and interop](#adapter-format-and-interop)
- [Limitations](#limitations)

## Training on the quantized model

`gmlx train` fine-tunes a quantized GGUF model as it is, and writes the
adapter as a small GGUF file. The model is never converted, and
`--adapter` applies the result at load.

The usual way to fine-tune is to convert the model to full precision,
train, and quantize again. Training on the quantized model skips the
conversion and the second quantization. The model weights stay in their
quantized form and only the adapter trains, so gmlx keeps no
full-precision copy of the model and no optimizer state for it. You can therefore fine-tune a model that would
not fit in memory at full precision. At inference, the model file is not
changed, and the output is the quantized model plus the adapter at full
precision.

If you have the full-precision model and enough memory, fine-tune that
and quantize afterwards. Training on the quantized model is for when the
quantized model is all that fits.

## Train an adapter

Teaching Qwen3-0.6B to talk like a pirate shows the whole flow. The
example uses a dense Q8_0 model, which is the tested case. The trainer
accepts the other GGUF codecs and plain MLX models too. Download the
model into the folder of your [configuration file](config.md), which
registers it as `qwen3-0.6b-q8`:

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
```

`--data` takes a folder with `train.jsonl`, and optionally `valid.jsonl`,
in any format that the mlx-lm LoRA trainer accepts. The formats are chat
records of the form `{"messages": [...]}`, prompt and completion pairs, and
plain text. Chat records suit an instruct model, because the trainer
applies the chat template of the model. The loss covers the prompt as
well as the reply. `--data` also takes a Hugging Face dataset id, which
needs the `datasets` package in the environment of gmlx.

The example dataset,
[GPT007/pirate_speak](https://huggingface.co/datasets/GPT007/pirate_speak),
has 100 conversations as Llama-3-formatted text. This script writes them as
chat records, and it needs the `datasets` package:

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

Then train. The adapter covers every linear layer in the top
`--num-layers` layers of the model:

```sh
pip install datasets && python prep_pirate.py
gmlx train qwen3-0.6b-q8 --data ./pirate-data \
    --iters 150 --batch-size 4 --num-layers 8 --adapter-out ~/models/pirate-lora.gguf
```

Before it loads the model, `train` checks the data folder and that it can
write the adapter file. At the end it prints
`[gmlx] wrote N-module LoRA adapter -> PATH`. The training loss should
fall steadily. With only 90 examples, stop at about 150 iterations,
because longer training overfits the data, and the model then starts to
repeat itself.

`--num-layers` and `--rank` trade what the adapter can learn against
memory, and the defaults of 8 layers at rank 8 are a good start. When a
longer `--max-seq-length` runs out of memory, `--grad-checkpoint`
computes each layer again in the backward pass instead of keeping it,
which costs time. It needs `--dropout 0`, and Kimi K3 and DeepSeek-V4.1
refuse it. The flags are listed under [gmlx train](cli.md#gmlx-train).

## Use the adapter

```sh
gmlx run qwen3-0.6b-q8 --adapter ~/models/pirate-lora.gguf --prompt "What's the weather like today?"
gmlx run qwen3-0.6b-q8 --prompt "What's the weather like today?"    # The model without the adapter.
```

At load, gmlx prints `[adapter] applied N-module GGUF LoRA from ...`.
Qwen3 is a thinking model, so `run` prints its reasoning first. The pirate
data has no reasoning, so the adapted model thinks briefly and then
answers like a pirate.

In chat, `/adapter off` and `/adapter on` turn the adapter off and on for
the next turns, and `/adapter 0.5` scales it. `gmlx serve model.gguf
--adapter pirate-lora.gguf` serves the adapted model under an id derived
from the model file name, and the model without the adapter as
`<id>-base`, with no configuration file.

## Serving one base with many adapters

A server can offer one model under several adapters at once. The model
loads a single time, and each adapter loads into a slot of its own on that
model. A request applies only the adapter of the id that it names.
Requests to the base and to adapted ids batch together in one decode step,
and switching between ids swaps nothing.

Ids whose entries have the same `path` and differ only in `adapter`
share one loaded model:

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

The server finds a relative adapter path in
[`server.model_dirs`](config.md#servermodel_dirs), like a model path. All
settings that change how the model loads must match across the group,
because an id that differs in more than `adapter` gets a separate copy of
the model. [Memory and residency](config.md#memory-and-residency) lists
these settings, and a shared [profile](config.md#profiles) keeps them the
same. `GET /v1/metrics` shows the group as one entry under
`resident_models`, and `GET /v1/models` lists all three ids.

A request names an id, and nothing else in the API concerns adapters:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3-0.6b-pirate",
  "messages": [{"role": "user", "content": "Summarize RAID levels."}]
}'
```

Requests to different ids of a group do not wait for each other. An
adapted request gives the same output as it would alone, whatever else is
in the batch. To compare adapters in one conversation, run
`gmlx chat --server qwen3-0.6b-pirate` and switch with
[`/model <id>`](chat.md#undo-retry-and-sessions), which keeps the
conversation.

Adapters work with the prompt cache and speculative decoding:

- The key of a [prompt cache](config.md#prompt-cache) entry includes the
  adapter, so each id caches its own copy of a shared prefix.
- `speculative: true` works with adapters. Set it on every id of the group,
  because a difference splits the group. Adapters give the same output
  whether the batch uses speculation or not.

The adapters of a group are part of what identifies its loaded model. When
you add an id with a new adapter and reload the configuration, the server
builds a new copy, and the old copy unloads after its idle timeout. Plan
for both copies in memory for a short time, or restart the server instead
when the model is large.

## Adapter format and interop

The adapter file uses the llama.cpp GGUF LoRA format, which
`convert_lora_to_gguf.py` writes from a PEFT folder. The file has
`general.type` set to `adapter`, `adapter.type` set to `lora`, and
`adapter.lora.alpha`. Each adapted weight has a `lora_a` and `lora_b`
tensor pair named after the tensor of the base model.

gmlx therefore loads PEFT adapters converted with that script, and the
GGUF adapters that people publish for llama.cpp. Adapters from
`gmlx train` and `gmlx distill train` load in llama.cpp with `--lora`, for
the architectures that llama.cpp supports. `gmlx validate` recognizes an
adapter file.

## Limitations

- Only LoRA adapters work. DoRA on a quantized model is not supported.
- An adapter can change the dense linear layers and the down projections
  of MoE experts. Loading refuses an adapter that changes the gate or up
  projections of experts, the embeddings, or any expert weight on a model
  that runs a fused MoE block, such as gemma and gpt-oss.
- An adapter works on text models only, so `--adapter` does not combine
  with `--mmproj`.
- The adapter of each id is fixed at load. To use another adapter, a
  request names another id.
- The adapter must be for the architecture of the model. The loader checks
  this, and its error names both architectures. It does not check that the
  model is the fine-tune the adapter was trained on.
