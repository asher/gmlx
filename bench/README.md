# bench/ -- the harness behind docs/benchmarks.md

Everything needed to reproduce (or extend) the published benchmarks: the
serve-path A/B harness, the SVG chart renderer, and a merge tool for folding
partial reruns. Pure stdlib besides `gmlx` itself and `requests`;
`--chat-dataset` additionally needs `datasets`.

| File | What it does |
|---|---|
| `serve-bench.py` | gmlx vs llama.cpp (or ds4-server) through each server's OpenAI API: prefill + decode at KV depth, baseline + speculative (MTP) arms. Writes a md report, a raw JSON, and the same SVG charts the docs publish. |
| `plot-bench.py` | Stdlib SVG renderer for the three published chart grammars: `panels`, `fleet-ratio`, `mtp-lift`. Invoked automatically by `serve-bench.py` (`--no-svg` to skip). |
| `merge.py` | Folds many `serve-bench-*.json` into per-model files, newest cell wins. Use after partial reruns, then re-render charts from the merged files. |
| `example.json` | Config template (three models: native-MTP, drafter-MTP, no MTP). |
| `patches/ds4-ignore-eos.patch` | Required on the ds4-server arm for forced-length decode. |

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
with `--gmlx-bin` / `--llama-server-bin` / `--ds4-bin`.

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
- **Thermal alternation**: which runtime runs first alternates per round, with
  a cooldown (and optional die-temperature gate, `cool_to_c`) between server
  blocks; the median across rounds is reported.
- **Own-tokenizer accounting**: token counts are re-tokenized with the model's
  tokenizer, never taken from the server's self-reported usage.

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
  "thinking": null,                  // null = model default; see docstring
  "models": [
    {
      "name": "my-model-q6k",
      "gguf": "~/models/model.gguf",
      "tokenizer": "gguf",           // "gguf" = synthesized from the GGUF
                                     // metadata (recommended); or an HF id
      "kv_bits": null,               // quantized KV, symmetric both engines
      "mtp": {                       // omit for baseline-only
        "gmlx":  { "speculative": true },        // native MTP head, or:
                                                 // { "draft_gguf": "..." }
        "llama": { "spec_default": true },       // native head (+ngram), or:
                                                 // { "draft_gguf": "..." },
                                                 // spec_default:false isolates
                                                 // the head from the ngram
                                                 // speculator
        "ds4":   { "gguf": "...", "margin": 3.0 }
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
