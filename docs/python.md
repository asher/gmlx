# Python API

The `gmlx` package loads and runs GGUF models from your own Python code.
Its stable surface is the set of names that the package root exports in
`gmlx.__all__`.

Exports resolve lazily, so `import gmlx` returns immediately and never
imports MLX, which makes it safe in tooling that only inspects metadata. The
MLX and kernel-extension import happens at first use, and a broken runtime
environment fails at that point with a message naming the missing
component.

## Load a model

```python
from gmlx import load_model

model, config, tokenizer = load_model("model.gguf")
```

The model is a stock mlx-lm `Model` with its quantized leaves swapped for
`KQuant*` modules, so `mlx_lm.generate`, `mlx_lm.stream_generate` and any
other code written for ordinary mlx-lm checkpoints run it unchanged. The
config and tokenizer are both synthesized from the GGUF metadata. Sharded
files, named `-00001-of-000NN.gguf`, are discovered from any shard's path.

| Kwarg | Default | Meaning |
|---|---|---|
| `arch` | Detected | Override `general.architecture` detection. |
| `hf_source` | `None` | Load the config from this local directory or Hugging Face repo instead of synthesizing it from the GGUF metadata. |
| `chat_template` | From the GGUF | An inline Jinja string, or a path to a `.jinja`/`.txt` file, replaces the GGUF's chat template. |
| `target_prefix` | `""` | This prefix is added to every remapped tensor name. |
| `no_remap` | `False` | Skip the GGUF-to-HF tensor-name remap, which suits inspection but not inference. |
| `fail_on_unknown` | `False` | Raise `RuntimeError` on a tensor that has no remap entry, instead of skipping it with a warning. |
| `zero_copy` | `True` | Load tensors as no-copy mmap views. `False` copies into fresh buffers. |
| `verbose` | `False` | Print load diagnostics such as `[arch]`, `[gguf]` and `[patch]`. |

`load_model` takes no vision or drafter arguments. Pairing a model with an
mmproj file and speculative decoding are CLI and server features, as
[Vision and audio](vlm.md) and
[Speculative decoding](speculative-decoding.md) describe.

## Generate

```python
from gmlx import generate

text = generate(model, tokenizer, "Explain KV caching.", max_tokens=256)
```

A string prompt goes through the tokenizer's chat template when one is
present, while a pre-tokenized `list[int]` prompt is used as-is. The return
value is the generated text.

Sampling takes these keyword arguments:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_tokens` | `64` | Generation stops after this many tokens. |
| `temp` | `0.0` | It sets the sampling temperature, and `0.0` is greedy. |
| `top_p` | `0.95` | It sets the nucleus sampling threshold. |
| `top_k` | `0` | Sampling keeps only the k most likely tokens, and `0` disables the cutoff. |
| `min_p` | `0.05` | Sampling drops tokens less likely than this fraction of the top token. |
| `xtc_probability` / `xtc_threshold` | `0.0` | They set XTC sampling, which is active when the probability is nonzero. |
| `repetition_penalty` | `0.0` | It applies a classic repetition penalty over the last `repetition_context_size` tokens, `20` by default. `0.0` turns it off. |
| `presence_penalty` / `frequency_penalty` | `0.0` | They apply OpenAI-style penalties. |
| `logit_bias` | `None` | Each `{token_id: bias}` entry is added to the logits. |
| `stop` | `None` | Generation ends when one of these strings appears, and the match is trimmed. |

The rendering of the prompt follows these keyword arguments:

| Kwarg | Default | Meaning |
|---|---|---|
| `apply_chat_template` | `True` | Set `False` for base models or pre-templated text. |
| `system_prompt` | `None` | It is prepended as a system message on the templated path. |
| `template_kwargs` | `None` | These extra kwargs go to `apply_chat_template`, such as `{"enable_thinking": False}`. |

The KV cache is set by these keyword arguments:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_kv_size` | `None` | Cap the KV cache with a rotating window. |
| `kv_bits` | `None` | Quantize the KV cache to this many bits. |
| `kv_group_size` | `64` | It sets the KV quantization group size. |
| `quantized_kv_start` | `0` | KV quantization begins at this position. |
| `kv_quant_scheme` | `None` | It selects `uniform` for the standard affine scheme or `kvarn` for variance-normalized quantization. |
| `kv_tail_tokens` | `1024` | Under `kvarn`, this many recent tokens also stay fp16. It is a multiple of 128, and `0` disables the tail. |

The remaining keyword arguments control long prompts, thinking models and output:

| Kwarg | Default | Meaning |
|---|---|---|
| `prefill_step_size` | Model-aware | It sets the prefill chunk width, and the default follows the deployed choice for the model. |
| `prefill_progress` | `False` | Show a stderr spinner during a long prefill, on a TTY only, cleared before the first token. |
| `thinking_budget` | `None` | Cap reasoning tokens. After about N thinking tokens a `</think>` is forced so the model answers. A model that never opens `<think>` is unaffected. |
| `thinking_start_token` / `thinking_end_token` | `None` | They set the reasoning markers for a model whose markers are not detected from its tokenizer or template. The end tag is the one the budget forces. |
| `verbose` | `False` | Stream text and timing to stdout while generating. |
| `reasoning` | `None` | It sets how a verbose stream shows thinking. `show` styles it, `hide` folds it into the timing line and `raw` streams it. The return value is always raw. |

## Benchmark

```python
from gmlx import bench

bench(model, tokenizer, lengths=(512, 4096, 16384))
# {512: {"prefill_tps": ..., "decode_tps": ...}, 4096: {...}, ...}
```

`bench` measures prefill and decode throughput at each prompt length through
the real generation path, with chunked prefill and the async one-step-ahead
decode pipeline. The numbers therefore match deployed throughput, not a
naive forward loop. Its CLI equivalent is `gmlx run --bench`.

| Kwarg | Default | Meaning |
|---|---|---|
| `lengths` | `(512, 4096, 16384)` | `bench` sweeps these prompt lengths. |
| `decode_tokens` | `32` | Each run measures this many decode tokens. |
| `runs` | `2` | `bench` runs each length this many times and reports the best. |
| `warmup` | `True` | An untimed warmup generation runs first. |
| `prefill_step_size` | Model-aware | It works as in `generate`. |
| `kv_bits` | `None` | It works as in `generate`. |
| `kv_group_size` | `64` | It works as in `generate`. |
| `quantized_kv_start` | `0` | It works as in `generate`. |
| `kv_quant_scheme` | `None` | It works as in `generate`. |
| `kv_tail_tokens` | `1024` | It works as in `generate`. |

## Preflight and errors

```python
from gmlx import UnsupportedCodecError, UnsupportedArchError
from gmlx.load.preflight import preflight

pf = preflight("model.gguf")
pf.arch, pf.shards, pf.codec_histogram, pf.n_tensors, pf.n_params
```

`preflight(gguf_path, *, arch=None, hf_source=None)` checks a GGUF before
a load. It finds the shards, counts the tensors of each codec, refuses
unsupported codecs by name and checks the architecture. It reads only the
GGUF header, so a file of many GB is checked in well under a second.
`load_model` runs it internally, and you can call it yourself to check a
file first. Its CLI equivalent is `gmlx validate`.

`preflight` and `load_model` raise the same exceptions:

- `UnsupportedCodecError` means a tensor codec with no kernel here. It
  carries `.arch` and `.unsupported`, a `{codec: count}` dict.
- `UnsupportedArchError` means a GGUF architecture the loader cannot build
  a model for.
- `HadamardFoldError`, from `gmlx.load.preflight`, means a Hadamard-folded
  file whose fold version or architecture the loader does not support.
- `FileNotFoundError` means missing shards of a split file, and
  `ValueError` means a truncated file or a header without an architecture.

`ARCH_TABLE` maps each supported GGUF architecture id to its runtime entry,
with the fields `gguf_arch`, `model_type`, `family`, `remap_alias`,
`notes`, `backend` and `caveat`. [Supported architectures](arch-coverage.md)
is the generated view of the same data for people to read, with the
validation status of each entry.

## Tokenizer without the model

```python
from gguf import GGUFReader
from gmlx import detect_arch, load_tokenizer_from_gguf

reader = GGUFReader("model.gguf", "r")
arch = detect_arch(reader)
tokenizer = load_tokenizer_from_gguf(reader, arch)
```

`load_tokenizer_from_gguf(meta, arch, *, chat_template_override=None)`
builds an HF fast tokenizer from the GGUF's embedded vocab, merges and
scores metadata. It is the same synthesis `load_model` runs, taken on its
own. `detect_arch(reader)` reads `general.architecture` from the header.
Neither touches tensor bytes.

These suit a tool that needs the tokenizer before it decides whether to
load weights, such as an eval harness that checks tokenizer parity,
pre-tokenizes a corpus or inspects a template. `chat_template_override`,
an inline Jinja string, replaces the GGUF's chat template. The loader
infers the model's turn-ending tokens from the template it ends up with, so
an override changes which tokens stop generation as well.

Three helpers read a tokenizer's vocabulary as bytes, for tools that line
up two tokenizers over the same text, such as `gmlx distill align`. They
take an HF fast tokenizer or an mlx-lm tokenizer wrapper.

```python
from gmlx import token_bytes, whitespace_start_mask, vocab_map_hash

tb = token_bytes(tokenizer)                    # list[bytes | None], one per id
ws = whitespace_start_mask(tokenizer, len(tb), tb)   # bool array, True at space-initial ids
key = vocab_map_hash(tokenizer)                # 16 hex digits over the id-to-token map
```

`token_bytes(tokenizer, width=None)` returns the byte string of every id
below `width`, which defaults to the vocabulary size. Specials and unfilled
ids are `None`. ByteLevel vocabularies go through the GPT-2 byte decoder,
and SentencePiece vocabularies map the U+2581 marker to a space and
`<0xNN>` pieces to that byte. `whitespace_start_mask(tokenizer, width,
token_bytes_list=None)` marks the ids whose bytes start with ASCII
whitespace plus the end-of-sequence ids. `vocab_map_hash(tokenizer)`
hashes the id-to-token map with specials left out. Equal hashes mean
equal maps, not identical tokenization, since merges, the pre-tokenizer
and the normalizer are not covered.

## mlx-lm server bridge

```python
from gmlx import install_gguf_bridge

install_gguf_bridge()
```

`install_gguf_bridge` patches `mlx_lm.server.ModelProvider` so that any
`*.gguf` model path loads through `load_model`, and a second call changes
nothing. Other paths pass through unchanged, so one `mlx_lm.server` process
can mix GGUF files and ordinary MLX checkpoints. GGUF requests are pinned
to mlx-lm's sequential path, with no batching. A `--draft-model` is
ignored for GGUF models with a warning, and `--adapter` on one raises,
since adapters are wired only in `gmlx serve`. Use the bridge to add GGUF
support to an existing `mlx_lm.server` deployment, and `gmlx serve` for
everything else.

## Quantized modules

The swapped leaves are `KQuantLinear`, `KQuantEmbedding`,
`KQuantSwitchLinear` and `KQuantMultiLinear`, the canonical classes in
`mlx_kquant.nn`, re-exported here. Each stores the GGUF file bytes directly
as a `uint8` `weight` and dispatches through the `mlx_kquant` Metal kernels
on a stock `mlx` wheel, so dequantization happens inside the kernel, never
as a separate materialized pass.

`install_kquant_modules(model, hf_kquant_meta)` is the swap step. It visits
the leaf modules of a constructed model and replaces each one whose weight
carries a codec. Because it keys on codec strings instead of the
architecture, a custom loader can use it on any model.

## Beyond the stable surface

Other modules are importable but internal, among them the VLM loader,
embeddings and rerank, the CPU-offload paths and the server. Their
signatures change without notice. `generate` also accepts experimental
parameters that are not part of the stable surface. To have an internal
piece made public, open an issue.
