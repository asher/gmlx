# bench/ -- the harness behind docs/benchmarks.md

Everything needed to reproduce (or extend) the published benchmarks: the
serve-path A/B harness, the SVG chart renderer, and a merge tool for folding
partial reruns. Pure stdlib besides `gmlx` itself and `requests`;
`--chat-dataset` additionally needs `datasets`.

| File | What it does |
|---|---|
| `serve-bench.py` | gmlx vs llama.cpp (or ds4-server, oMLX or MTPLX) through each server's OpenAI API: prefill + decode at KV depth, baseline + speculative (MTP) arms. Writes a md report, a raw JSON, and the same SVG charts the docs publish. |
| `plot-bench.py` | Stdlib SVG renderer for the published chart grammars: `panels`, `fleet-ratio`, `mtp-lift`, `batch-scaling`. The first three are invoked automatically by `serve-bench.py` (`--no-svg` to skip); `batch-scaling` (aggregate decode vs concurrent streams, optional `--ref` overlay of a second run) is rendered by hand from a concurrency-sweep JSON. |
| `merge.py` | Folds many `serve-bench-*.json` into per-model files, newest cell wins. Use after partial reruns, then re-render charts from the merged files. |
| `example.json` | Config template (three models: native-MTP, drafter-MTP, no MTP). |
| `patches/ds4-ignore-eos.patch` | Required on the ds4-server arm for forced-length decode. |
| `omlx-serve.py` | Runs `omlx serve` with forced-length decode for the oMLX arm. |
| `patches/mtplx-ignore-eos.patch` | Required on the MTPLX arm for forced-length decode. |
| `repack-rec-mlx.py` | Rewrites a RecursiveSI Rec checkpoint, such as Flint's, into the layout oMLX loads. |

## Quickstart

```sh
# one GGUF, default depth/concurrency ladder -- ALWAYS dry-run first
./serve-bench.py ~/models/model.gguf --tokenizer gguf --dry-run
./serve-bench.py ~/models/model.gguf --tokenizer gguf

# a config sweep (example.json pins the instruct-MTP chat corpus, see below)
./serve-bench.py --config example.json
```

Outputs land in `results/`: `serve-bench-<stamp>.md` (report),
`serve-bench-<stamp>.json` (raw samples), and per-model `*-panels.svg` +
`fleet-ratio.svg` + `mtp-lift.svg` (light and dark variants), matching the
grammar of the charts in `docs/benchmarks.md`.

Binaries resolve from `PATH` (`gmlx`, `llama-server`, `ds4-server`); override
with `--gmlx-bin` / `--llama-server-bin` / `--ds4-bin`. The oMLX arm runs the
python of the venv that has omlx, `--omlx-python`. The MTPLX arm runs the
mtplx of its venv, `--mtplx-bin`.

## Fairness guards (why these numbers are comparable)

The full details live in the `serve-bench.py` docstring; the short version:

- **Same GGUF** feeds both runtimes (gmlx is zero-conversion).
- **Caches off** on both sides, plus a unique nonce prefixed to every prompt,
  so no prefix/KV cache can ever hit.
- **Forced-length decode** (`--ignore-eos` both sides) so decode tok/s is
  measured over an identical token window.
- **Identical sampling body** sent to both runtimes (default: temp 0.6,
  top-p 0.95, top-k 20 -- a realistic deploy regime, never greedy, which
  distorts MTP acceptance).
- **Thermal alternation**: server blocks run in a fixed order and odd rounds
  reverse it, with a cooldown (and optional die-temperature gate, `cool_to_c`)
  between blocks. The median across rounds is reported.
- **Own-tokenizer accounting**: prompt and output tokens are counted with the
  model's tokenizer. A reply that ends on its length limit counts the server's
  `completion_tokens`, or `max_tokens` when the server reports none, because
  some servers drop think tags and special tokens from the streamed text.

## Config schema

All keys optional except `models`; defaults in `serve-bench.py --help` / the
`load_specs` docstring.

```jsonc
{
  "chat_dataset": "HuggingFaceH4/ultrachat_200k",
                                     // instruct-MTP prompt corpus, pinned in
                                     // the config so runs are reproducible;
                                     // CLI --chat-dataset overrides. Optional
                                     // chat_split (train_sft) /
                                     // chat_max_convs (8000) alongside
  "depths": [512, 4096, 16384],      // KV depth ladder (prompt tokens)
  "concurrency": [1],                // parallel streams per cell
  "draft_depths": [3],               // MTP draft-token counts to sweep
  "max_tokens": 192,                 // forced decode length
  "rounds": 2,                       // thermal-alternated repetitions
  "warmup": 1,                       // unmeasured warmup requests per block
  "cooldown": 20.0,                  // fixed seconds between server blocks
  "cool_to_c": 50.0,                 // optional die-temp gate (max wait below)
  "cool_max_wait": 240.0,
  "llama_ctx_cap": null,             // clamp llama -c (set to n_ctx_train for
                                     // very deep cells; avoids RoPE scaling)
  "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "seed": 1234 },
                                     // optional reasoning_effort is sent
                                     // to every server as is
  "thinking": null,                  // null = model default; see docstring
  "shared_prompts": false,           // true: every model and arm gets the same
                                     // prompts, and the report adds a table
                                     // of all arms
  "models": [
    {
      "name": "my-model-q6k",
      "gguf": "~/models/model.gguf",
      "tokenizer": "gguf",           // "gguf" = synthesized from the GGUF
                                     // metadata (recommended); or an HF id
      "kv_bits": null,               // quantized KV, symmetric both engines
      "omlx_model": null,            // MLX checkpoint for --vs omlx; optional
                                     // omlx_args append `omlx serve` flags
      "mtplx_model": null,           // MLX pack for --vs mtplx; optional
                                     // mtplx_args append `mtplx serve` flags
      "runtimes": null,              // runtimes this model runs on (null = all)
      "draft_depths": null,          // this model's own draft depths
      "concurrency": null,           // this model's own concurrency list
      "mtp_only": false,             // skip this model's baseline arm
      "mtp": {                       // omit for baseline-only
        "gmlx":  { "speculative": true },        // native MTP head, or:
                                                 // { "draft_gguf": "..." }
        "llama": { "spec_default": true },       // native head (+ngram), or:
                                                 // { "draft_gguf": "..." },
                                                 // spec_default:false isolates
                                                 // the head from the ngram
                                                 // speculator
        "ds4":   { "gguf": "...", "margin": 3.0 },
        "mtplx": { "enabled": true }
      }
    }
  ]
}
```

(JSON does not allow comments -- this block is annotated schema documentation;
start from `example.json`.)

## MTP measurement notes

- **Instruct/chat targets need a chat corpus** (`chat_dataset` config key or
  `--chat-dataset`, e.g. `HuggingFaceH4/ultrachat_200k`): acceptance is
  corpus-dependent, and real multi-turn chat sent as a `messages` array keeps
  the draft head on-distribution at every depth. Raw text (`--corpus`) is for
  base/continuation models; the embedded fallback corpus is unrepresentative
  and only fit for smoke tests.
- **Never compare MTP cells across thinking modes**: acceptance is strongly
  content-sensitive (thinking-mode text drafts much better than no-think
  answers). Leave `thinking` at `null` (model default) unless you are
  equalizing a ds4 A/B.
- Draft depth is equalized across engines (gmlx `--draft-block-size N+1` ==
  llama `--spec-draft-n-max N`).

## Partial reruns

Rerun any subset (`--only gmlx`, a single depth, one model) into the same
results tree, then fold and re-render:

```sh
./merge.py results merged
./plot-bench.py panels merged/my-model-q6k.json --model my-model-q6k \
    --drop-depth 0 --out merged/my-model-q6k-panels.svg
```

Newest cell wins per `model|runtime|arm|depth|concurrency`, so untouched
cells persist. Directories whose name starts with `_tainted` or
`_out-of-scope` are excluded from merging.

## The ds4-server arm

`--vs ds4` benches DeepSeek-V4 GGUFs against dwarfstar's ds4-server instead
of llama.cpp. Build ds4-server with `patches/ds4-ignore-eos.patch` applied
(see the patch header), and read the ds4 notes in the `serve-bench.py`
docstring: sampling equalization interacts with ds4's thinking mode, ds4 MTP
arms require greedy sampling, and concurrency > 1 measures queueing rather
than batching there.

## The oMLX arm

`--vs omlx` benches gmlx against oMLX. oMLX loads MLX checkpoints, so each
model names an MLX checkpoint of the same weights as `omlx_model`, and the
gmlx arm still runs the GGUF.

Install oMLX 0.7.2 or later into its own venv from a checkout of its source,
with its custom kernels built:

```sh
uv venv -p 3.12 --seed ~/.venvs/omlx
make -C path/to/omlx install PYTHON=~/.venvs/omlx/bin/python
```

Add `omlx_model` to each model in the config:

```json
{
  "name": "flint-int4-q4_1-omlx",
  "gguf": "~/models/flint-int4-q4_1.gguf",
  "tokenizer": "gguf",
  "omlx_model": "~/models/flint-int4-mlx"
}
```

Then point `--omlx-python` at that venv:

```sh
./serve-bench.py --config flint-omlx.json --vs omlx \
    --omlx-python ~/.venvs/omlx/bin/python --dry-run
```

`omlx-serve.py` makes every reply run to `max_tokens`, as on the other
engines. The oMLX arm has no speculative arms.

### Flint checkpoints

RecursiveSI's [flint-int4](https://huggingface.co/RecursiveSI/flint-int4)
is stored for their Rec runtime, which oMLX cannot load. `repack-rec-mlx.py`
rewrites it into the stock MLX layout, with the tensor bytes unchanged:

```sh
./repack-rec-mlx.py ~/models/flint-int4 ~/models/flint-int4-mlx
```

## The MTPLX arm

`--vs mtplx` benches gmlx against [MTPLX](https://github.com/youssofal/mtplx)
by Youssof Altoukhi, a native MTP speculative decoding server for Apple
Silicon. MTPLX loads its own MLX packs, so each model on the MTPLX arm names
one as `mtplx_model`. With `"mtp": {"mtplx": {"enabled": true}}` the model
gets MTP arms, and draft depth N runs `mtplx serve --mtp --depth N`. The
server runs with its SSD session cache off, and `"thinking": false` runs it
with `--reasoning off`.

Install MTPLX into its own venv, then apply `patches/mtplx-ignore-eos.patch`
to it as its header shows. The patch is made against mtplx 2.12.2.

```sh
uv venv -p 3.13 ~/.venvs/mtplx
uv pip install -p ~/.venvs/mtplx/bin/python mtplx==2.12.2
```

The patch adds a switch that makes every reply run to `max_tokens`, as on the
other engines. `serve-bench.py` turns the switch on and refuses an mtplx
without the patch. Pass the venv's mtplx as `--mtplx-bin`.

MTPLX and gmlx run different files and different drafters, so a fair
comparison gives each engine its own model entry. These per-model keys shape
the arms of each entry:

- `runtimes` lists the runtimes the model runs on.
- `draft_depths` replaces the top-level draft depths for the model.
- `concurrency` replaces the top-level concurrency list for the model.
- `mtp_only` skips the model's baseline arm.
- An entry with `mtplx_model` and no `gguf` can name the GGUF as its
  `tokenizer`, so that every arm counts tokens with the same tokenizer.

Set `"shared_prompts": true` so that every model and arm gets the same
prompts. The report then adds a table of all arms.

This config reproduces the published Qwen3.8-27B comparison. gmlx runs
Qwen3.8-27B-Q4_0-pure.gguf once with the DFlash2 drafter
Qwen3.8-27B-DFlash2-Q4_0.gguf from
[incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
and once with the model's own MTP head. MTPLX runs the pack
[Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed).
Each engine runs at its own best draft depth.

Make the GGUF from
[quark75/Qwen3.8-27B-Q4_0.gguf](https://huggingface.co/quark75/Qwen3.8-27B-Q4_0.gguf)
by converting its one Q6_K tensor, the output projection, to Q4_0:

```sh
llama-quantize --allow-requantize --pure Qwen3.8-27B-Q4_0.gguf \
    Qwen3.8-27B-Q4_0-pure.gguf Q4_0
```

```json
{
  "chat_dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K",
  "chat_split": "train",
  "chat_max_convs": 8000,
  "depths": [512, 4300, 17000],
  "concurrency": [1],
  "draft_depths": [3],
  "max_tokens": 256,
  "rounds": 2,
  "warmup": 1,
  "cooldown": 20.0,
  "cool_to_c": 50.0,
  "cool_max_wait": 240.0,
  "thinking": null,
  "shared_prompts": true,
  "sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "seed": 1234, "reasoning_effort": "medium" },
  "models": [
    {
      "name": "gmlx-dflash-b8",
      "gguf": "~/models/Qwen3.8-27B-Q4_0-pure.gguf",
      "tokenizer": "gguf",
      "runtimes": ["gmlx"], "mtp_only": true, "draft_depths": [7],
      "mtp": { "gmlx": { "draft_gguf": "~/models/Qwen3.8-27B-DFlash2-Q4_0.gguf" } }
    },
    {
      "name": "gmlx-mtp-b6",
      "gguf": "~/models/Qwen3.8-27B-Q4_0-pure.gguf",
      "tokenizer": "gguf",
      "runtimes": ["gmlx"], "mtp_only": true, "draft_depths": [5],
      "mtp": { "gmlx": { "speculative": true } }
    },
    {
      "name": "mtplx-bare",
      "mtplx_model": "~/models/Qwen3.8-27B-MTPLX-Bare-Speed",
      "tokenizer": "~/models/Qwen3.8-27B-Q4_0-pure.gguf",
      "runtimes": ["mtplx"], "mtp_only": true, "draft_depths": [3],
      "mtp": { "mtplx": { "enabled": true } }
    }
  ]
}
```

The published run used 8 requests per cell at these depths. The deep cells
used depths 67000, 110000 and 200000 with 2 requests per cell:

```sh
./serve-bench.py --config mtplx-qwen38-27b.json --vs mtplx \
    --mtplx-bin ~/.venvs/mtplx/bin/mtplx --requests 8 --startup-timeout 900 --dry-run
```

gmlx draft depth N runs `--draft-block-size N+1`, so `draft_depths` [7] is
DFlash2 with block size 8 and [5] is the MTP head with block size 6. The
report takes MTPLX acceptance from the `timings` of each response.
