# Python API

The `gmlx` package loads and runs GGUF models from your own Python code:

```python
from gmlx import load_model, generate

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain KV caching.", max_tokens=256))
```

The stable API is the set of names in `gmlx.__all__`, plus `preflight` and
`HadamardFoldError` from `gmlx.load.preflight`. `import gmlx` is fast and
does not import MLX, so tools that only read metadata can use it. MLX loads
at the first call that needs it.

- [Load a model](#load-a-model)
- [Generate](#generate)
- [Benchmark](#benchmark)
- [Preflight and errors](#preflight-and-errors)
- [Tokenizer without the model](#tokenizer-without-the-model)
- [mlx-lm server bridge](#mlx-lm-server-bridge)
- [Quantized modules](#quantized-modules)

## Load a model

`load_model("model.gguf")` returns `(model, config, tokenizer)`. The model
is a stock mlx-lm model, so `mlx_lm.generate`, `mlx_lm.stream_generate` and
other mlx-lm code run it unchanged. The config and tokenizer come from the
GGUF metadata. For a split file, named
`-00001-of-000NN.gguf`, pass the path of any shard.

| Kwarg | Default | Meaning |
|---|---|---|
| `arch` | Detected | Overrides `general.architecture`. |
| `hf_source` | `None` | A local folder or Hugging Face repo to load the config from, instead of the GGUF metadata. |
| `chat_template` | From the GGUF | An inline Jinja string, or a path to a `.jinja` or `.txt` file. |
| `target_prefix` | `""` | A prefix added to every remapped tensor name. |
| `no_remap` | `False` | Keeps the GGUF tensor names. For inspection, not inference. |
| `fail_on_unknown` | `False` | Raises `RuntimeError` on a tensor with no remap entry, instead of skipping it with a warning. |
| `zero_copy` | `True` | Loads tensors as views of the mapped file. `False` copies them. |
| `verbose` | `False` | Prints load diagnostics. |

Vision projectors and speculative decoding are not available here. They
work in the CLI and the server, as [Vision and audio](vlm.md) and
[Speculative decoding](speculative-decoding.md) describe.

## Generate

`generate(model, tokenizer, prompt, ...)` returns the generated text. A
string prompt goes through the model's chat template, and a `list[int]`
prompt is used as it is.

Sampling:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_tokens` | `64` | The most tokens to generate. |
| `temp` | `0.0` | Sampling temperature. `0.0` is greedy. |
| `top_p` | `0.95` | Nucleus sampling threshold. |
| `top_k` | `0` | Keeps only the k most likely tokens. `0` turns it off. |
| `min_p` | `0.05` | Drops tokens less likely than this fraction of the top token. |
| `xtc_probability` / `xtc_threshold` | `0.0` | XTC sampling, on when the probability is above 0. |
| `repetition_penalty` | `0.0` | Repetition penalty over the last tokens. `0.0` turns it off. |
| `repetition_context_size` | `20` | How many recent tokens the repetition penalty looks at. |
| `presence_penalty` / `frequency_penalty` | `0.0` | OpenAI-style penalties. |
| `logit_bias` | `None` | A `{token_id: bias}` dict added to the logits. |
| `stop` | `None` | Strings that end generation. The match is trimmed. |

The KV cache:

| Kwarg | Default | Meaning |
|---|---|---|
| `max_kv_size` | `None` | Caps the KV cache with a rotating window. |
| `kv_bits` | `None` | Quantizes the KV cache to this many bits. |
| `kv_group_size` | `64` | KV quantization group size. |
| `quantized_kv_start` | `0` | Starts quantizing once the cache holds this many tokens. |
| `kv_quant_scheme` | `None` | `uniform` for the standard scheme, or `kvarn` for variance-normalized quantization. |
| `kv_tail_tokens` | `1024` | Under `kvarn`, recent tokens kept at fp16. A multiple of 128. `0` turns it off. |

The prompt, thinking and output:

| Kwarg | Default | Meaning |
|---|---|---|
| `apply_chat_template` | `True` | `False` sends the text as it is, for base models and pre-templated text. |
| `system_prompt` | `None` | A system message added before the prompt. |
| `template_kwargs` | `None` | Extra chat-template variables, such as `{"enable_thinking": False}`. |
| `prefill_step_size` | Model-aware | Prefill chunk width. The default is the one gmlx picks for the model. |
| `prefill_progress` | `False` | Shows a spinner on stderr during a long prefill, on a terminal only. |
| `thinking_budget` | `None` | Ends the reasoning after about this many tokens, so the model answers. |
| `thinking_start_token` / `thinking_end_token` | `None` | Reasoning markers for a model whose markers are not detected. |
| `verbose` | `False` | Streams text and timing to stdout. |
| `reasoning` | `None` | How the verbose stream shows thinking: `show`, `hide` or `raw`. The return value is always raw. |

## Benchmark

```python
from gmlx import bench

bench(model, tokenizer, lengths=(512, 4096, 16384))
# {512: {"prefill_tps": ..., "decode_tps": ...}, 4096: {...}, ...}
```

`bench` measures prefill and decode speed at each prompt length through the
same path that generation uses, so the numbers match real use. The CLI
equivalent is `gmlx run --bench`.

| Kwarg | Default | Meaning |
|---|---|---|
| `lengths` | `(512, 4096, 16384)` | The prompt lengths to measure. |
| `decode_tokens` | `32` | Decode tokens per run. |
| `runs` | `2` | Runs per length. The best is reported. |
| `warmup` | `True` | Runs an untimed generation first. |

`bench` also takes `prefill_step_size`, `kv_bits`, `kv_group_size`,
`quantized_kv_start`, `kv_quant_scheme` and `kv_tail_tokens`, with the same
meaning and defaults as in `generate`.

## Preflight and errors

```python
from gmlx import UnsupportedCodecError, UnsupportedArchError
from gmlx.load.preflight import preflight

pf = preflight("model.gguf")
pf.arch, pf.shards, pf.codec_histogram, pf.n_tensors, pf.n_params
```

`preflight(gguf_path, *, arch=None, hf_source=None)` checks a GGUF before
you load it. It finds the shards, counts the tensors of each codec, and
checks the codecs and the architecture. It reads only the header, so it
takes well under a second. `load_model` runs it too. The CLI equivalent is
`gmlx validate`.

Both raise the same exceptions:

| Exception | Meaning |
|-----------|---------|
| `UnsupportedCodecError` | A tensor codec gmlx has no kernel for. `.unsupported` is a `{codec: count}` dict, and `.arch` names the architecture. |
| `UnsupportedArchError` | A GGUF architecture gmlx cannot build. |
| `HadamardFoldError` | A Hadamard-folded file whose fold version or architecture is not supported. From `gmlx.load.preflight`. |
| `FileNotFoundError` | Missing shards of a split file. |
| `ValueError` | A truncated file, or a header without an architecture. |

`ARCH_TABLE` maps each supported GGUF architecture id to its entry, with
the fields `gguf_arch`, `model_type`, `family`, `remap_alias`, `notes`,
`backend` and `caveat`. [Supported architectures](arch-coverage.md) shows
the same data.

## Tokenizer without the model

To get the tokenizer without loading any weights, read the GGUF header:

```python
from gguf import GGUFReader
from gmlx import detect_arch, load_tokenizer_from_gguf

reader = GGUFReader("model.gguf", "r")
arch = detect_arch(reader)
tokenizer = load_tokenizer_from_gguf(reader, arch)
```

- `detect_arch(reader)` reads `general.architecture`.
- `load_tokenizer_from_gguf(meta, arch, *, chat_template_override=None)`
  builds the same HF fast tokenizer that `load_model` does. An override
  template also changes which tokens end a turn.

Three helpers read a vocabulary as bytes, for tools that line up two
tokenizers, such as `gmlx distill align`. Each takes an HF fast tokenizer
or an mlx-lm tokenizer wrapper:

```python
from gmlx import token_bytes, whitespace_start_mask, vocab_map_hash

tb = token_bytes(tokenizer)                          # list[bytes | None], one per id
ws = whitespace_start_mask(tokenizer, len(tb), tb)   # bool array
key = vocab_map_hash(tokenizer)                      # 16 hex digits
```

- `token_bytes(tokenizer, width=None)` gives each id's bytes, or `None` for
  specials and unused ids.
- `whitespace_start_mask(tokenizer, width, token_bytes_list=None)` marks
  ids that start with whitespace, and end-of-sequence ids.
- `vocab_map_hash(tokenizer)` hashes the id-to-token map. Equal hashes mean
  equal maps, not equal tokenization.

## mlx-lm server bridge

```python
from gmlx import install_gguf_bridge

install_gguf_bridge()
```

`install_gguf_bridge` makes `mlx_lm.server` load any `*.gguf` path through
`load_model`, and leaves other paths alone. Use it to add GGUF files to an
existing `mlx_lm.server`. GGUF requests run one at a time, `--draft-model`
is ignored for them, and `--adapter` raises. Use `gmlx serve` for
everything else.

## Quantized modules

`load_model` replaces the model's quantized layers with `KQuantLinear`,
`KQuantEmbedding`, `KQuantSwitchLinear` and `KQuantMultiLinear`, from
`mlx_kquant.nn`. Each keeps the GGUF bytes as its `weight` and dequantizes
inside the Metal kernel.

`install_kquant_modules(model, hf_kquant_meta)` does that swap. It replaces
each layer whose weight has a codec, whatever the architecture, so a custom
loader can use it on any model.

Other modules, such as the VLM loader and the server, are internal and
change without notice. So do the experimental `generate` arguments that
this page does not list.
