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
This page covers chat models. Image models take LoRAs too, as
[Use a LoRA](images.md#use-a-lora) describes.

- [Training data](#training-data)
- [Train an adapter](#train-an-adapter)
- [Use the adapter](#use-the-adapter)
- [Serving one base with many adapters](#serving-one-base-with-many-adapters)
- [Mixing adapters per request](#mixing-adapters-per-request)
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

`--adapter PATH:STRENGTH` applies the adapter at a strength, 1 when left
out. Strength 0.5 adds half the adapter's change, 2 adds twice, and a
negative strength subtracts it. `serve`, `run`, `chat`, `capture` and
`kld` read the flag the same way: the text after the last colon is a
strength only when it is a number, so a path that holds a colon needs no
suffix. Repeat the flag to apply more adapters.

```sh
gmlx run qwen3-0.6b-q8 --adapter ~/models/pirate-lora.gguf:0.5 --prompt "What's the weather like today?"
```

In `gmlx chat`, the command-line strength applies from the first turn.
A strength is absolute, never a multiple of another:

| Command | Effect |
|---------|--------|
| `/adapter` | Print the strength in effect. |
| `/adapter 0.5` | Run the adapter at strength 0.5 from the next turn. |
| `/adapter on` | Go back to the command-line strength. |
| `/adapter off` | Run the adapter at strength 0. |

`gmlx serve` with `--adapter` needs no config file. It serves the adapted
model under an id from the model's file name, here `qwen3-0.6b`, and the
model without the adapter as `qwen3-0.6b-base`. The strength is the
default for requests that set none, as an entry's
[`adapters`](#mixing-adapters-per-request) strength is:

```sh
gmlx serve ~/models/unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --adapter ~/models/pirate-lora.gguf:0.8
```

To measure how far an adapter moves the model at a few strengths, run
[`gmlx kld`](cli.md#gmlx-kld) with one `--adapter` flag for each strength.

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
names:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3-0.6b-pirate",
  "messages": [{"role": "user", "content": "What is the weather like today?"}]
}'
```

Requests to different ids batch together and do not wait for each other.
To keep one shared copy:

- Keep every other setting the same across the ids, including
  `speculative`. An id that differs in more than its adapters loads its own
  copy. A shared [profile](config.md#profiles) keeps the settings equal.
- Write a relative adapter path as you would a model path. It resolves in
  [`server.model_dirs`](config.md#servermodel_dirs).
- Adding an adapter and reloading the config builds a new copy, and the old
  one unloads after its idle timeout. For a large model, restart the server
  instead.

To check the sharing, look at `resident_models` in `/v1/metrics`. One
entry whose `ids` lists all three ids means one shared copy. Two entries
mean two copies:

```sh
curl -s http://127.0.0.1:8080/v1/metrics | jq '.server.resident_models[] | {ids, footprint_bytes}'
```

To compare adapters in one conversation, run
`gmlx chat --server qwen3-0.6b-pirate` and switch with
[`/model <id>`](chat.md#undo-retry-and-sessions).

The adapter files count in the copy's memory at twice their size, beside
the model, when the server checks whether the model fits.

## Mixing adapters per request

An entry can load several adapters, each at a default strength. List them
in [`adapters`](config.md#modelsadapters). A bare path runs at strength 1:

```yaml
models:
  qwen3-0.6b-mix:
    path: unsloth__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
    adapters:
      - {path: pirate-lora.gguf, strength: 0}
      - {path: formal-lora.gguf, strength: 0.5}
```

A request sets the strength of any of these adapters for itself only.
Requests at different strengths batch together, and a strength change does
not reload the model. The chat, completions, Responses and Messages routes
take two fields, and so do `/v1/systemone` and `/v1/prewarm`:

| Field | Shape | Adapters the request does not list |
|-------|-------|-------------------------------------|
| `adapters` | `[{"id": "pirate-lora.gguf", "strength": 0.8}]`, or a bare id for strength 1 | Keep the entry's default strength |
| `lora` | `[{"id": 0, "scale": 0.8}]`, as in llama-server | Run at 0. An empty list keeps the defaults. |

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3-0.6b-mix",
  "adapters": [{"id": "pirate-lora.gguf", "strength": 0.8}],
  "messages": [{"role": "user", "content": "What is the weather like today?"}]
}'
```

An `adapters` id is one of these:

- The adapter's path as the config writes it, such as `loras/pirate-lora.gguf`.
- The file name, with or without its extension, when only one adapter of
  the entry has it.
- A path inside a model folder that is the same file as one of the
  entry's adapters.

A `lora` id is the adapter's index in the entry's `adapters` list, from 0.
`/v1/models` lists each entry's adapters with their `index`, file `name`,
default `strength` and `sha256`. The `sha256` is the hash of the bytes the
loaded copy read, and it is `null` while the model is not loaded.

A request gets a 400 in these cases:

- It sets both `adapters` and `lora`.
- An id is an absolute path, starts with `~`, or has a `..` part.
- An id names no adapter of the entry, including one of another entry
  on the same model, or a file name that two adapters share.
- An id is listed twice, or a strength is not a number from -100 to 100.
- The entry loads no adapters and a field is not empty.

A request cannot add an adapter the entry does not load.

`lora` differs from llama-server in three ways: a bad or repeated id gets a
400, a `lora` item needs its `scale`, and the server has no
`/lora-adapters` route for global strengths.

## Adapter format and interop

Adapters use the llama.cpp GGUF LoRA format. gmlx loads PEFT adapters
converted with llama.cpp's `convert_lora_to_gguf.py` and the GGUF adapters
published for llama.cpp, and llama.cpp loads gmlx adapters with `--lora`.
An adapter with no `adapter.lora.alpha`, or an alpha of 0, applies at its
strength with no alpha scaling, as in llama.cpp.

A PEFT adapter that trains `embed_tokens` holds a `token_embd.weight`
pair. gmlx adds its delta to each looked-up row before the model scales the
rows, where llama.cpp adds it. `gmlx train` writes no such pair.

## Limitations

- Only LoRA adapters work. DoRA on a quantized model is not supported.
- An adapter can change the dense linear layers, the down projections of
  MoE experts and the token embedding. Loading refuses an adapter that
  changes the gate or up projections of experts. It also refuses expert
  weights on gpt-oss and on native-FP expert stacks, which
  `GMLX_FUSED_MOE=0` loads unfused.
- Loading refuses a token embedding pair before the model loads on a model
  that uses its token embedding as the output head, such as Gemma 4, on a
  Hadamard-rotated embedding and on Granite Switch. llama.cpp cannot apply
  the pair on these models either. It also refuses a
  `per_layer_token_embd.weight` pair, which llama.cpp loads but never
  applies.
- On a model whose MoE blocks run fused, such as Qwen3.5 and Qwen3.6 MoE, a
  block runs its slower unfused path only in a step where a request has one
  of the block's adapters on. In `gmlx serve` the other steps keep the
  fused speed. `gmlx run`, its `--bench` included, and `gmlx chat` run
  every step of an adapted block unfused.
- An adapter that changes expert weights on such a model needs the experts
  in memory. A load that streams the experts refuses it, with
  `stream: experts`, `--stream-experts`, `--stream-cpu`, or a model too
  large for memory. Set `GMLX_FUSED_MOE_BLOCK=0` to load the blocks unfused
  and stream them.
- Adapters work on text models only, so `--adapter` does not combine with
  `--mmproj`.
- `gmlx run` and `gmlx chat` apply an entry's adapters at the entry's
  strengths. In `gmlx chat`, `/adapter S` sets every adapter to S.
- A drafter for speculative decoding uses the model's output head and
  token embedding without an adapter on them, as llama.cpp's drafters do.
  Only the draft ignores an `output.weight` or `token_embd.weight` pair,
  which can lower the share of accepted drafts. The verify step uses the
  adapters, so the output is the same.
- An image model that borrows a chat model as its text encoder runs the
  encoder with every adapter off.
- `--grad-checkpoint` is refused on Kimi K3 and DeepSeek-V4.1.
- The loader checks that the adapter is for the model's architecture, but
  not that the model is the one it was trained on.
