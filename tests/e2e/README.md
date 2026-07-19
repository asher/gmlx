# Server end-to-end test harness

A black-box regression harness for the gmlx server. It launches the **real** entry
point (`python -m gmlx.server`) across a matrix of start modes and config features,
fires a fixed prompt suite plus per-feature post-checks at each live server, and grades
every response for errors, incoherence, and regressions.

It is deliberately **not** part of the pytest unit suite — every scenario loads a model
and most need the GPU. The detectors underneath it (`checks.py`) *are* unit-tested, by
`tests/test_e2e_checks.py`. Files here are not `test_`-prefixed, so pytest skips them.

## What it covers

Each **scenario** is one server launch + the requests fired at it + post-checks on the
live server. Scenarios are grouped into tiers:

| tier | what it exercises |
| --- | --- |
| `core` | single positional GGUF; config profiles, `extends`, `@profile`, `system`, aliases, unknown-id 404 / unknown-profile 400 |
| `kv` | quantized KV cache (baseline / 8-bit / 4-bit) under a deep planted-fact recall + long generation |
| `cache` | APC prompt cache: disabled / memory-only / SSD disk tier / disk × 8-bit-KV combo; reuse must be byte-identical |
| `residency` | multi-model LRU eviction under a tight weight-byte budget; idle-TTL reaping (pinned exempt) |
| `template` | chat-template override (config profile + single-model `--chat-template`); distinct templates fork distinct resident entries |
| `endpoints` | `/v1/models`, `/v1/metrics` resident view, `/unload` (one + all), `/v1/reload`, cache reset; plus the SSE streaming path |
| `negative` | the HF-download gate refuses a stray non-GGUF id while offline |
| `discovery` | `--models-dir` header-only scan serves derived ids |
| `vlm` | gemma-4-E2B + mmproj describes an image |
| `mtp` | gemma-4-E2B + assistant drafter (speculative) stays coherent; lossless-greedy spec == base |

A scenario whose required models aren't present under the models root is **skipped**,
not failed.

The coherence-judged tiers (`core`, `kv`, `template`, `endpoints`) run on **gemma-4-E2B**,
not a 0.6B model: a 0.6B model is too weak to follow strict-format prompts, so the judge
correctly fails it on baseline weakness — noise that masks real config regressions. E2B
gives a clean baseline where a config-induced degradation stands out. The structural tiers
(`residency`, `discovery`, `negative`) keep the tiny models — they need distinct small
sizes for LRU eviction and only fire the easy `capital` prompt. The **`cache` tier also
stays on Qwen3-0.6B**: APC block/prefix reuse needs a plain `KVCache`, and gemma-4's
sliding-window `RotatingKVCache` bypasses APC entirely, so only an APC-capable model can
verify the cache feature at all. Thinking is disabled on every request so a reasoning
preamble can't pollute the judged text (a target can re-enable it via `sampling`).
Deterministic single-fact prompts (`capital`, `math`, `multiturn`) skip the LLM judge —
their anchor is the correctness oracle, so the judge there is pure false-negative risk.

## Grading: two layers

1. **Floor checks** (`checks.py`, pure stdlib, deterministic) — transport/schema,
   `finish_reason`, usage, non-empty, no mojibake (U+FFFD), no NaN/inf flood, and a
   repetition/looping detector. Every successful response must clear the floor.
2. **LLM-as-judge** (`judge.py`) — a local model rules on *semantic* coherence
   (fluent, on-topic, not subtly looping) that a regex can't catch. It runs as a
   decoupled final phase, after every server is torn down, so it never shares the GPU
   with a server subprocess. A parse failure is a soft pass (judge flakiness can't fail
   a clean response); a clear "incoherent / repeating / low score" is a hard fail.

Depth matters: the suite includes a long planted-fact ("needle") context and a long
generation, because degeneration usually surfaces at depth, not on a one-liner.

## Running it

Use the project interpreter (the one with `gmlx` + `mlx_kquant` installed).

```bash
# CPU only: build + validate the whole matrix (YAML round-trips through the loader),
# load no model. Always run this first after touching the harness or config schema.
python tests/e2e/run_server_e2e.py --dry-run

# Print the plan (model inventory + which scenarios will run) and exit.
python tests/e2e/run_server_e2e.py --list

# Full run (GPU). Writes report.md + report.json under a fresh temp dir.
python tests/e2e/run_server_e2e.py

# A subset of tiers, with an explicit output dir.
python tests/e2e/run_server_e2e.py --tiers core,kv,cache --out ./e2e-out

# Skip the LLM judge (floor checks only — faster, fully deterministic).
python tests/e2e/run_server_e2e.py --no-judge

# Re-grade a prior run's responses with the judge, without re-launching servers.
python tests/e2e/run_server_e2e.py --judge-only ./e2e-out/report.json
```

Key flags: `--tiers` (comma list or `all`), `--filter SUBSTR` (scenario-key substring),
`--models-root DIR`, `--judge-model GGUF` (default: best available local), `--quick`
(short prompt suite for the generic core target), `--image PATH` (VLM tier; defaults to
the bundled `assets/cats.jpg`, override with `--image` or `$GMLX_E2E_IMAGE`),
`--out DIR`, `--python INTERP` (server subprocess interpreter).

The VLM tier uses a real photo (`assets/cats.jpg`) by default, which actually exercises
the vision encoder; a synthesized shapes PNG is only a last-resort backstop if no real
image resolves.

## Output

Under `--out` (or a temp dir printed at the end):

- `report.md` — human-readable: summary, per-tier roll-up, a Failures section, and a
  per-scenario breakdown (post-checks, each request's floor/anchor/judge verdicts, a
  response snippet). Rewritten incrementally so a long run is inspectable as it goes.
- `report.json` — the same data, machine-diffable across commits.
- `configs/<key>.yaml` — the exact config each scenario served.
- `logs/<key>.log` — the server subprocess stdout/stderr (argv + env header).
- `fixtures/` — generated template files, discovery symlink dirs, APC disk dirs.

Exit code is non-zero if any non-skipped scenario failed.

## Models

Resolved by logical handle against the models root (`--models-root`, default per
`models.py`). Handles + candidate filenames live in `models.py`; adjust them there to
match a given machine's library. **gemma-4-E2B** is the workhorse for the judged tiers
(core/kv/cache/template/endpoints) and the VLM/MTP tiers (VLM adds its mmproj; MTP adds
the assistant drafter GGUF). The small dense models (Qwen3-0.6B Q4/Q8, gemma-3-1B) back
only the structural tiers (residency LRU/TTL, discovery, the HF-gate negative) — they
need distinct small sizes for eviction. The judge prefers a larger coherent model
(gemma-4-12B) and falls back to the small ones.

A scenario whose models aren't present is **skipped**, so the harness runs on a partial
library — the structural + small-model tiers light up as soon as the public models are on
disk, even before the gemma-4 tiers.

### Getting the models

```bash
# Print copy-paste `gmlx pull` lines for every model not yet under --models-root
# (CPU, no network — it only inspects the filesystem and prints).
python tests/e2e/run_server_e2e.py --print-pull
```

The small dense models (Qwen3-0.6B Q4/Q8, gemma-3-1B) have public GGUF sources wired into
`models.py` `_SOURCES`, so `--print-pull` emits a ready-to-run line for each:

```bash
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ~/llm/gguf/qwen3-0.6b
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf   --to ~/llm/gguf/qwen3-0.6b
gmlx pull hf:ggml-org/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf --to ~/llm/gguf/gemma-3-1b-it-GGUF
```

The **gemma-4 family** (E2B judged/VLM/MTP companions, the 12B judge) has no public GGUF
distribution, so `--print-pull` prints the exact path to drop each file at instead. Either
provide your own gemma-4 GGUFs there, set their `hf:` ref in `_SOURCES`, or just run the
non-gemma-4 tiers — the gemma-4 scenarios skip cleanly when absent. The pull commands land
each file exactly where `models.py` looks for it, so a subsequent `--list` shows it present.

## Layout

| file | role |
| --- | --- |
| `run_server_e2e.py` | orchestrator: phases 0–3, argparse, report writing |
| `run_lora_e2e.py` | focused runner: GGUF LoRA train → serve → assert the adapter shifts output |
| `run_apc_disk_e2e.py` | focused runner: disk-backed APC (`APC_DISK_PATH`) populates from purely sequential single-user traffic, survives a server restart, works under multi-client batching, and is namespace-isolated per model |
| `scenarios.py` | the config matrix — one `Scenario` per feature/combination |
| `prompts.py` | the prompt suite (short / instruct / system / needle / long-gen / vlm) |
| `checks.py` | deterministic floor detectors (unit-tested separately) |
| `judge.py` | the LLM-as-judge (decoupled final phase) |
| `models.py` | handle → on-disk path registry |
| `client.py` | stdlib HTTP client (never raises on HTTP error status) |
| `server_proc.py` | launch / poll `/health` / tear down a server subprocess |
| `report.py` | result model + Markdown/JSON rendering |
