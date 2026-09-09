# Contributing

Thanks for considering a contribution. This page covers the mechanics. For
design context, the docs under [docs/](docs/) are authoritative.

## Dev setup

gmlx needs Python 3.11 or newer on macOS with Apple Silicon, which is the
primary target. `mlx-kquant` comes from PyPI as a prebuilt arm64 wheel for
macOS 26.2 and newer, and older macOS builds it from source, which needs
full Xcode with its Metal toolchain. It also builds CPU-only on Linux,
which is enough for the default test tier. The version bounds on mlx-vlm,
mlx-lm, mlx-kquant and mlx, and why they are what they are, are explained in
[docs/internals/upstream-upgrades.md](docs/internals/upstream-upgrades.md).

Dev setup is a venv, a clone and an editable install with the same extras
CI uses. Without `assistant`, the MCP tool-server tests skip themselves:

```sh
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/asher/gmlx
pip install -e "./gmlx[chat,assistant]" pytest ruff
```

## Tests

There are three tiers, described in full in
[docs/internals/testing.md](docs/internals/testing.md):

```sh
pytest                                   # CPU logic tests, no models, runs anywhere
KQUANT_TEST_GGUF_DIR=~/llm/gguf pytest   # adds numerical parity against real GGUFs, and -m integration runs only those
python tests/e2e/run_server_e2e.py       # server end-to-end harness, needs the GPU
```

A PR should keep the default `pytest` tier passing, and if your change
touches loading or numerics, say which integration tests you ran and on
which model. A new architecture has its own acceptance gate, including
long-context parity against llama.cpp and a regenerated coverage matrix,
in [docs/internals/adding-architectures.md](docs/internals/adding-architectures.md).

## Lint

```sh
ruff check .
python scripts/check-docs.py   # docs style and link check, also a CI step
pre-commit install             # optional, runs ruff on each commit
```

## Things to know before you patch

- The serving stack is stock mlx-vlm with late-bound patches over its
  seams, in `gmlx/serve/bridge_vlm.py`, `gmlx/serve/residency.py` and
  `gmlx/serve/patches/`. The loader patches a few mlx-lm classes at load
  time, and `gmlx/serve/bridge_lm.py` separately patches the
  `ModelProvider._load` of `mlx_lm.server`, the sequential mlx-lm server.
  Every patch is registered as a seam, guarded, and raises rather than
  no-ops when upstream moves. The rules for adding one and for moving the
  pins are in
  [docs/internals/upstream-upgrades.md](docs/internals/upstream-upgrades.md).
- Each concern has a module. Tensor-name remap is in `gmlx/load/remap.py`,
  config synthesis in `gmlx/load/config_synth.py` and arch metadata in
  `gmlx/load/arch_table.py`. A new architecture usually touches exactly those
  three plus a parity test.
- The package tree follows subsystems. Tests mirror it under `tests/`:

  | Package | Concern |
  |---------|---------|
  | `gmlx/load/` | GGUF discovery, parsing, remap, config synthesis, model construction |
  | `gmlx/models/` | owned model backbones, one subpackage or module per family |
  | `gmlx/upstream/` | patches installed over upstream mlx-lm/mlx-vlm seams |
  | `gmlx/cache/` | automatic prompt cache and KV-cache persistence |
  | `gmlx/spec/` | speculative decoding, MTP, drafters, acceptance |
  | `gmlx/stream/` | weight streaming and residency for over-RAM models |
  | `gmlx/serve/` | server, admission, batched decode, with HTTP patches in `serve/patches/` |
  | `gmlx/gen/` | generation loop, sampling profiles, benchmarks |
  | `gmlx/commands/` | CLI verbs behind the `gmlx` umbrella |
  | `gmlx/tui/` | interactive chat terminal UI |
  | `gmlx/talk/` | voice client, audio I/O and hotkey |
  | `gmlx/assistant/` | tool-loop assistant brain and its MCP surface |

  Cross-cutting modules such as `config.py`, `envflags.py`,
  `eval_guard.py`, `textfmt.py` and `spinner.py` stay at the `gmlx/` top
  level.
- Error messages name the fix. Follow the existing style. Say what was
  expected, what was found and what the user or upgrader should do.

## Commit style

A commit message is a single subject line with no body, in the form
`type(scope): short lowercase summary`. The type is one of `feat`, `fix`,
`perf`, `docs`, `test` and `chore`, plus `release` for a version bump. The
scope in parentheses names the subsystem or model family the change is
about, such as `stream`, `kv`, `server`, `cli` or `qwen4exp`, and is left
out when the change has no single home, as in `docs: fix audit findings`.
Examples from the history are `feat(arch): add falcon-h1`,
`fix(kv): honor kv_bits under MTP on cache-list models` and
`release: 0.4.10`. A revert is `chore(scope): revert <what>`.
