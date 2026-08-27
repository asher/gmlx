# Contributing

Thanks for considering a contribution. This page covers the mechanics; for design
context, the docs under [`docs/`](docs/) are the source of truth.

## Dev setup

`mlx-kquant` is on PyPI with prebuilt arm64 wheels for Python 3.10-3.14 on
macOS 26+. Older macOS builds it from source, which needs the Xcode Command
Line Tools. It pins `mlx==0.31.2` itself, so nothing else needs pinning. Dev
setup is a venv, a clone, and an editable install:

```sh
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/asher/gmlx
pip install -e "./gmlx[chat]" pytest ruff
```

macOS on Apple Silicon is the primary target. mlx-kquant also builds CPU-only on
Linux, which is enough for the default test tier.

## Tests

Three tiers (full guide: [docs/testing.md](docs/testing.md)):

```sh
pytest                                   # CPU logic tests: no models, runs anywhere
KQUANT_TEST_GGUF_DIR=~/llm/gguf pytest   # + numerical parity vs real GGUFs (add -m integration to run only those)
python tests/e2e/run_server_e2e.py       # server end-to-end harness (GPU)
```

A PR should keep `pytest` (the default tier) green. If your change touches
loading/numerics, say which integration tests you ran and on which model. New
architectures need a greedy token-parity check against llama.cpp at long context,
and must keep `scripts/check-coverage.py --check --strict` green with
`docs/arch-coverage.md` regenerated. Short-prompt parity is not sufficient:
attention bugs only surface at depth.
What adding an architecture involves, and the full acceptance gate:
[docs/adding-architectures.md](docs/adding-architectures.md).

## Lint

```sh
ruff check .
pre-commit install   # optional: runs the same check on each commit
```

## Things to know before you patch

- Seam patches are version-fragile by design. The serving stack adopts mlx-vlm's
  FastAPI app + batching engine by patching late-bound seams (`gmlx/serve/bridge_vlm.py`,
  `gmlx/serve/residency.py`, `gmlx/serve/patches/`), and the loader patches a few
  mlx-lm classes at load time. `gmlx/serve/bridge_lm.py` separately patches `mlx_lm.server`'s
  `ModelProvider._load` (the sequential mlx-lm server, not mlx-vlm). Every patch
  carries a guard or version tripwire that fails loudly.
  Keep that property: a new patch must be idempotent and must raise (not silently
  no-op) when the upstream surface it expects has changed. The `mlx-vlm` upper
  bound in `pyproject.toml` is bumped deliberately, after re-running the server
  tests against the new version.
- One module per concern: tensor-name remap lives in `gmlx/load/remap.py`,
  config synthesis in `gmlx/load/config_synth.py`, arch metadata in
  `gmlx/load/arch_table.py`. A new architecture usually touches exactly those
  three plus a parity test.
- The package tree maps subsystems; tests mirror it under `tests/`:

  | Package | Concern |
  |---------|---------|
  | `gmlx/load/` | GGUF discovery, parsing, remap, config synthesis, model construction |
  | `gmlx/models/` | owned model backbones, one subpackage or module per family |
  | `gmlx/upstream/` | patches installed over upstream mlx-lm/mlx-vlm seams |
  | `gmlx/cache/` | automatic prompt cache and KV-cache persistence |
  | `gmlx/spec/` | speculative decoding: MTP, drafters, acceptance |
  | `gmlx/stream/` | weight streaming and residency for over-RAM models |
  | `gmlx/serve/` | server, admission, batched decode; HTTP patches in `serve/patches/` |
  | `gmlx/gen/` | generation loop, sampling profiles, benchmarks |
  | `gmlx/commands/` | CLI verbs behind the `gmlx` umbrella |
  | `gmlx/tui/` | interactive chat terminal UI |
  | `gmlx/talk/` | voice client: audio I/O and hotkey |
  | `gmlx/assistant/` | tool-loop assistant brain and its MCP surface |

  Cross-cutting leaves (`config.py`, `envflags.py`, `eval_guard.py`,
  `textfmt.py`, `spinner.py`) stay at the `gmlx/` top level.
- Error messages name the fix. Follow the existing style: say what was
  expected, what was found, and what the user (or upgrader) should do.

## Commit style

A single line, no body: `(topic): short imperative summary`, e.g.
`(arch): add falcon-h1`, `(server): fix XTC 400 on bare-int eos_token_ids`.
The topic is parenthesized and names one top-level feature; pick from the
established set so history stays greppable:

`arch`, `loader`, `server`, `cli`, `chat`, `mtp`, `adapter`, `train`,
`stream` (formerly `cpu-moe`), `vlm`, `manage`, `launch`, `config`, `bench`,
`tests`, `docs`, `release`, `hygiene`.

Keep everything on the subject line, no extended body. A revert is
`(topic): revert <what>`.
