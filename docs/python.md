# Python API

The CLI is the primary interface. This page is for embedding gmlx in your own
Python. The stable surface is exactly what the package root exports as
`gmlx.__all__`. All of it is documented here.

Exports resolve lazily. `import gmlx` returns immediately and never imports
MLX, which makes it safe in tooling that only inspects metadata. The MLX and
kernel-extension import happens at first use. A broken runtime environment
fails at that point with a message naming the missing component.

## Load a model

```python
from gmlx import load_model

model, config, tokenizer = load_model("model.gguf")
```

The return value is `(model, config, tokenizer)`, where the model is a stock
mlx-lm `Model` with quantized leaves swapped for `KQuant*` modules, the config
is the dict synthesized from the GGUF metadata and the tokenizer is built from
the same metadata. Under `mlx_lm.generate` and `mlx_lm.stream_generate` the
model runs normally, in code written for ordinary mlx-lm checkpoints. Sharded
files, named `-00001-of-000NN.gguf`, are discovered from any shard's path.

| Kwarg | Default | Meaning |
|---|---|---|
| `arch` | detected | Override `general.architecture` detection. |
| `hf_source` | `None` | Load the config from this local dir or HF repo id instead of synthesizing it, for arches without a synthesizer or variants whose constants differ. |
| `chat_template` | from GGUF | Inline Jinja string, or a path to a `.jinja`/`.txt` file, replacing the GGUF's chat template. |
| `no_remap` | `False` | Skip the GGUF-to-HF tensor-name remap. For inspection, not inference. |
| `zero_copy` | `True` | Load tensors as no-copy mmap views. `False` copies into fresh buffers. |
| `verbose` | `False` | Print load diagnostics such as `[arch]`, `[gguf]` and `[patch]`. |

There are no vision or draft-model kwargs here. Pairing a model with an mmproj
file and speculative decoding are CLI and server features, covered in
[vlm.md](vlm.md) and
[performance.md](performance.md#mtp-speculative-decoding).

## Generate

```python
from gmlx import generate

text = generate(model, tokenizer, "Explain KV caching.", max_tokens=256)
```

A string prompt goes through the tokenizer's chat template when one is
present, while a pre-tokenized `list[int]` prompt is used as-is. The return
value is the generated text.

Sampling:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_tokens` | `64` | Generation cap. |
| `temp` | `0.0` | Temperature. `0.0` is greedy. |
| `top_p` | `0.95` | Nucleus sampling. |
| `top_k` | `0` | Top-k cutoff. `0` disables it. |
| `min_p` | `0.05` | Minimum-probability cutoff. |
| `xtc_probability` / `xtc_threshold` | `0.0` | XTC sampling, active when the probability is nonzero. |
| `repetition_penalty` | `0.0` | Classic repetition penalty over the last `repetition_context_size` tokens, default `20`. |
| `presence_penalty` / `frequency_penalty` | `0.0` | OpenAI-style penalties. |
| `logit_bias` | `None` | `{token_id: bias}` added to the logits. |
| `stop` | `None` | Stop strings. Generation ends when one appears. The match is trimmed. |

Prompt handling:

| Kwarg | Default | Meaning |
|---|---|---|
| `apply_chat_template` | `True` | Set `False` for base models or pre-templated text. |
| `system_prompt` | `None` | Prepended as a system message on the templated path. |
| `template_kwargs` | `None` | Extra `apply_chat_template` kwargs, such as `{"enable_thinking": False}`. |

KV cache:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_kv_size` | `None` | Cap the KV cache with a rotating window. |
| `kv_bits` | `None` | Quantize the KV cache to this many bits. |
| `kv_group_size` | `64` | KV quantization group size. |
| `quantized_kv_start` | `0` | Position where KV quantization begins. |

Long prompts and thinking models:

| Kwarg | Default | Meaning |
|---|---|---|
| `prefill_step_size` | model-aware | Prefill chunk width. The default follows the deployed choice for the model. |
| `prefill_progress` | `False` | Show a stderr spinner during a long prefill, on a TTY only, cleared before the first token. |
| `thinking_budget` | `None` | Cap reasoning tokens. After roughly N thinking tokens a `</think>` is forced so the model answers. No-op when the model never opens a `<think>` block. |
| `thinking_start_token` / `thinking_end_token` | `None` | Reasoning markers for a model whose markers are not detected from its tokenizer or template. The end tag is the one the budget forces. |
| `verbose` | `False` | Stream text and timing to stdout while generating. |

## Benchmark

```python
from gmlx import bench

bench(model, tokenizer, lengths=(512, 4096, 16384))
# {512: {"prefill_tps": ..., "decode_tps": ...}, 4096: {...}, ...}
```

`bench` measures prefill and decode throughput at each prompt length through
the real generation path, chunked prefill and the async one-step-ahead decode
pipeline. The numbers therefore match deployed throughput, not a naive forward
loop. Its CLI equivalent is `gmlx run --bench`.

| Kwarg | Default | Meaning |
|---|---|---|
| `lengths` | `(512, 4096, 16384)` | Prompt lengths to sweep. |
| `decode_tokens` | `32` | Decode window measured in each run. |
| `runs` | `2` | Runs at each length. The best is reported. |
| `warmup` | `True` | One untimed warmup generation first. |
| `prefill_step_size` | model-aware | As in `generate`. |

## Preflight and errors

```python
from gmlx import UnsupportedCodecError, UnsupportedArchError
from gmlx.load.preflight import preflight

pf = preflight("model.gguf")
pf.arch, pf.shards, pf.codec_histogram, pf.n_tensors, pf.n_params
```

`preflight(gguf_path, arch=None)` validates a GGUF before committing to a
load. It discovers shards, histograms the tensor codecs, refuses unsupported
ones by name and gates on the architecture. Because it reads only the GGUF
header, it stays fast on multi-GB files. `load_model` runs it internally. Call
it yourself to check a file first. Its CLI equivalent is `gmlx validate`.

Failures raise one of two exceptions, from `preflight` or `load_model`
alike:

- `UnsupportedCodecError` means a tensor codec with no kernel here. It
  carries `.arch` and `.unsupported`, a `{codec: count}` dict.
- `UnsupportedArchError` means a GGUF architecture the loader cannot build
  a model for.

`ARCH_TABLE` maps each supported GGUF architecture id to its runtime entry,
with the fields `gguf_arch`, `model_type`, `family`, `remap_alias`,
`notes`, `backend` and `caveat`. [arch-coverage.md](arch-coverage.md) is
the generated human-readable view of the same data, with validation
status.

## Tokenizer without the model

```python
from gguf import GGUFReader
from gmlx import detect_arch, load_tokenizer_from_gguf

reader = GGUFReader("model.gguf", "r")
arch = detect_arch(reader)
tokenizer = load_tokenizer_from_gguf(reader, arch)
```

`load_tokenizer_from_gguf(meta, arch, *, chat_template_override=None)` builds
an HF fast tokenizer purely from the GGUF's embedded vocab, merges and
scores metadata. It is the same synthesis `load_model` runs internally,
without the model load. `detect_arch(reader)` reads `general.architecture`
from the header. Both read only GGUF metadata, so they stay fast on
multi-GB files.

Use these when a tool needs the tokenizer before deciding whether to load
weights at all, such as an eval harness doing tokenizer parity checks,
corpus pre-tokenization, or template inspection. `chat_template_override`,
an inline Jinja string, replaces the GGUF's chat template. It is applied
before multi-EOS inference, so EOS detection runs against the override.

## mlx-lm server bridge

```python
from gmlx import install_gguf_bridge

install_gguf_bridge()
```

Idempotently patches `mlx_lm.server.ModelProvider` so any `*.gguf` model path
loads through `load_model`. Non-GGUF paths pass through unchanged, which lets
a single `mlx_lm.server` process mix GGUF and ordinary MLX checkpoints. GGUF
requests are pinned to mlx-lm's validated sequential path, with no batching.
Adapters and draft models are not supported on this route. Use it to add GGUF
support to an existing `mlx_lm.server` deployment. `gmlx serve` is the
full-featured server.

## Quantized modules

The swapped leaves are `KQuantLinear`, `KQuantEmbedding`, `KQuantSwitchLinear`
and `KQuantMultiLinear`, the canonical classes in `mlx_kquant.nn`, re-exported
here. Each stores the GGUF file bytes directly as a `uint8` `weight` and
dispatches through the `mlx_kquant` Metal kernels on a stock `mlx` wheel.
Dequantization happens inside the kernel, never as a separate materialized
pass. `install_kquant_modules(model, hf_kquant_meta)` is the swap step itself.
It iterates over a constructed model's leaf modules and replaces each one
whose weight carries a codec. Being driven entirely by codec strings, it is
arch-generic and exported for building custom loaders.

## Beyond the stable surface

Other modules are importable but internal, among them the VLM loader,
embeddings and rerank, the CPU-offload paths and the server. Their signatures
change without notice. `generate` additionally accepts experimental parameters
that are intentionally undocumented here. If you need an internal piece as a
public API, open an issue.
