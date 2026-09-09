# Contributing

Thanks for considering a contribution. This page covers the mechanics. For
design context, the docs under [docs/](docs/) are authoritative.

## Dev setup

`mlx-kquant` is on PyPI with prebuilt arm64 wheels for Python 3.10-3.14 on
macOS 26.2+. Older macOS builds it from source, which needs full Xcode with
its Metal toolchain. It pins `mlx==0.32.1`. Nothing else needs pinning. Dev
setup is a venv, a clone and an editable install:

```sh
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/asher/gmlx
pip install -e "./gmlx[chat]" pytest ruff
```

macOS on Apple Silicon is the primary target. mlx-kquant also builds CPU-only on
Linux, which is enough for the default test tier.

## Tests

There are three tiers, described in full in
[docs/internals/testing.md](docs/internals/testing.md):

```sh
pytest                                   # CPU logic tests, no models, runs anywhere
KQUANT_TEST_GGUF_DIR=~/llm/gguf pytest   # adds numerical parity against real GGUFs, and -m integration runs only those
python tests/e2e/run_server_e2e.py       # server end-to-end harness, needs the GPU
```

A PR should keep the default `pytest` tier passing. If your change touches
loading or numerics, say which integration tests you ran and on which
model. New architectures need a greedy token-parity check against llama.cpp
at long context. They must keep `scripts/check-coverage.py --check --strict`
passing with `docs/arch-coverage.md` regenerated. Short-prompt parity is
not sufficient, because attention bugs only appear at depth.
[docs/internals/adding-architectures.md](docs/internals/adding-architectures.md)
describes what adding an architecture involves and the full acceptance
gate.

## Lint

```sh
ruff check .
pre-commit install   # optional, runs the same check on each commit
```

## Things to know before you patch

- Seam patches are version-fragile by design. The serving stack uses
  mlx-vlm's FastAPI app and batching engine by patching late-bound seams in
  `gmlx/serve/bridge_vlm.py`, `gmlx/serve/residency.py` and
  `gmlx/serve/patches/`. At load time the loader patches a few mlx-lm
  classes. `gmlx/serve/bridge_lm.py` separately patches the
  `ModelProvider._load` of `mlx_lm.server`, the sequential mlx-lm server.
  Each patch has a guard or version check that raises an error. Keep that
  property. A new patch must be idempotent and must raise, never silently
  no-op, when the upstream surface it expects has changed. The `mlx-vlm`
  upper bound in `pyproject.toml` is bumped on purpose, after re-running
  the server tests against the new version.
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

A commit message is a single line with no body, in the form
`type(scope): short lowercase summary`. Examples are
`feat(arch): add falcon-h1` and
`fix(server): XTC 400 on bare-int eos_token_ids`. The type is one of
`feat`, `fix`, `perf`, `docs`, `test` and `chore`. A scope names a
subsystem and comes from the established set, which keeps history
greppable:

`arch`, `loader`, `server`, `cli`, `chat`, `mtp`, `adapter`, `train`,
`stream`, `vlm`, `manage`, `launch`, `config`, `bench`, `tests`, `docs`,
`release`, `hygiene`.

Keep everything on the subject line, no extended body. A revert is
`chore(scope): revert <what>`.
