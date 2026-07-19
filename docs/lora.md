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
rank 8) are a sensible start. Note the walkthrough targets dense bases; on a MoE
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

In config mode, `adapter:` is a per-model key. Because the adapter is part of a
model's load signature, `base`, `base+adapterA`, and `base+adapterB` are three
distinct model ids that can be resident side by side:

```yaml
models:
  qwen3-0.6b:
    path: Qwen3-0.6B-Q8_0.gguf
  qwen3-0.6b-pirate:
    path: Qwen3-0.6B-Q8_0.gguf
    adapter: pirate-lora.gguf
```

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
- Dense linears only (q/k/v/o, gate/up/down). An adapter targeting MoE expert
  stacks or embeddings errors loudly rather than being silently skipped.
- Text path only: `--adapter` doesn't combine with `--mmproj` (VLM) or
  `--speculative` (MTP) yet.
- One adapter per resident entry, fixed at load: switching adapters means
  addressing a different model id (see the config example above), not a
  per-request parameter.
- The adapter must match the base architecture (checked at load); matching the
  exact base finetune is your responsibility.
