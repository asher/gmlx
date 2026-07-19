# Contributing

Thanks for considering a contribution. This page covers the mechanics; for design
context, the docs under [`docs/`](docs/) are the source of truth.

## Dev setup

`mlx-kquant` is on PyPI with prebuilt arm64 wheels for Python 3.10-3.13 on
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
```

## Things to know before you patch

- Seam patches are version-fragile by design. The serving stack adopts mlx-vlm's
  FastAPI app + batching engine by patching late-bound seams (`server_bridge_vlm.py`,
  `residency.py`, `server_patches/`), and the loader patches a few mlx-lm classes
  at load time. `server_bridge_lm.py` separately patches `mlx_lm.server`'s
  `ModelProvider._load` (the sequential mlx-lm server, not mlx-vlm). Every patch
  carries a guard or version tripwire that fails loudly.
  Keep that property: a new patch must be idempotent and must raise (not silently
  no-op) when the upstream surface it expects has changed. The `mlx-vlm` upper
  bound in `pyproject.toml` is bumped deliberately, after re-running the server
  tests against the new version.
- One module per concern: tensor-name remap lives in `remap.py`, config
  synthesis in `config_synth.py`, arch metadata in `arch_table.py`. A new
  architecture usually touches exactly those three plus a parity test.
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
