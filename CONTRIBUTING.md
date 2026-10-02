# Contributing

A contribution to gmlx needs a dev setup, passing tests, clean lint and a
commit in the house style. Thanks for considering one. For design context,
the [gmlx documentation](docs/README.md) is authoritative.

## Dev setup

gmlx needs Python 3.11 or newer, and macOS 26.2 or newer on Apple Silicon.
Its kernel dependency, `mlx-kquant`, comes from PyPI as a prebuilt arm64
wheel. On Linux, `mlx-kquant` builds CPU-only, which is enough for the
default test tier. The version bounds on mlx-vlm, mlx-lm, mlx-kquant and mlx
are explained in
[Upgrading mlx-vlm, mlx-lm and mlx](docs/internals/upstream-upgrades.md).

Dev setup is a venv, a clone and an editable install with the `chat` and
`assistant` extras. Without `assistant`, the MCP tool-server tests skip
themselves. CI adds `vlm` as well, an empty extra kept for older install
commands. These commands create the dev setup:

```sh
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/asher/gmlx
pip install -e "./gmlx[chat,assistant]" pytest ruff
```

## Rust toolchain for the guest entry

Container mode runs a small static Linux program, the guest entry, from
`crates/gmlx-entry/`. The default test tier builds it for your Mac, so it
needs Rust from rustup with the Linux musl target. The crate's
`rust-toolchain.toml` names the stable channel, so rustup builds it with
the current release. Install rustup and the target once:

```sh
brew install rustup
export PATH="$(brew --prefix rustup)/bin:$PATH"
rustup toolchain install stable --profile minimal --target aarch64-unknown-linux-musl
```

With that `PATH`, `python scripts/build_guest_entry.py` builds the Linux
binary into `gmlx/container/guest/gmlx-entry`, and `--native` builds one
for the Mac. Every build checks the toolchain first. It prints these
commands when the musl target is missing, or when `cargo` or `rustc` is
older than the `rust-version` in the crate's `Cargo.toml`. A Homebrew
`rust` earlier on `PATH` has no musl target, so put rustup's folder first.
Builds use the vendored `libc` crate, so they need no network.

The same source and Rust release give identical bytes on one kind of
machine but not across macOS and Linux, because cargo hashes the build
machine into symbol names. A release therefore ships the binary that the
release job builds on arm64 Linux.

## Tests

There are three tiers, described in full in
[Testing](docs/internals/testing.md):

```sh
pytest                                   # CPU logic tests, no models, runs anywhere
KQUANT_TEST_GGUF_DIR=~/llm/gguf pytest   # adds numerical parity against real GGUFs, and -m integration runs only those
python tests/e2e/run_server_e2e.py       # server end-to-end harness, needs the GPU
```

A PR should keep the default `pytest` tier passing. If your change
touches loading or numerics, say which integration tests you ran and on
which model. A new architecture has its own acceptance gate and required
tests, in
[Adding a GGUF architecture](docs/internals/adding-architectures.md).

## Lint

```sh
ruff check .
python scripts/check-docs.py   # docs style and link check, also a CI step
pip install --group docs && mkdocs serve   # preview the docs site locally
pre-commit install             # optional, runs ruff on each commit
pip install "pyright[nodejs]==1.1.414" && pyright   # seam drift check, needs the deps installed
```

pyright guards against upstream symbol and signature drift, and it does not
type-check the whole tree. It covers the files in `[tool.pyright].include`, all
of them at zero errors. The other files that import mlx_vlm or mlx_lm
statically are listed in `[tool.gmlx.pyright].backlog` and join the gate
one file at a time as each reaches zero. `tests/test_pyright_scope.py`
holds both lists to the tree, so a new static import lands in one list or
the other.

A `# pyright: ignore[rule]` is allowed only where the false positive comes
from upstream typing (a stub that omits the attribute, a scalar union, a
field added to an upstream dataclass), with a comment saying which. Keep
the count under twenty repo-wide.

ruff's S110 rule flags a `try`/`except Exception: pass`. Either log the
failure, or say where the error goes on the except line:
`except Exception:  # noqa: S110 - <reason>`. A bare marker without a
reason is not accepted in review. Tests are exempt.

The docs site is built from `docs/` by MkDocs and published by the release
workflow for each release tag, so a docs change reaches the site with the
next release. A new page goes in the `nav` of `mkdocs.yml` as well as the
docs index. The site's CI build fails on a page that is missing from the
`nav`.

## Things to know before you patch

- The serving stack is stock mlx-vlm with late-bound patches over its
  seams, in `gmlx/serve/bridge_vlm.py`, `gmlx/serve/residency.py` and
  `gmlx/serve/patches/`. The loader patches a few mlx-lm classes at load
  time, and `gmlx/serve/bridge_lm.py` separately patches the
  `ModelProvider._load` of `mlx_lm.server`, the sequential mlx-lm server.
  Every patch is registered as a seam, guarded, and raises rather than
  no-ops when upstream moves. The rules for adding one and for moving the
  pins are in
  [Upgrading mlx-vlm, mlx-lm and mlx](docs/internals/upstream-upgrades.md).
- Each concern has a module. Tensor-name remap is in `gmlx/load/remap.py`,
  config synthesis in `gmlx/load/config_synth.py` and arch metadata in
  `gmlx/load/arch_table.py`. Those three are where a new architecture lands.
- The package tree follows subsystems. Tests mirror it under `tests/`:

  | Package | Concern |
  |---------|---------|
  | `gmlx/load/` | It covers GGUF discovery, parsing, remap, config synthesis and model construction. |
  | `gmlx/models/` | It holds the owned model backbones, one subpackage or module per family. |
  | `gmlx/upstream/` | It holds the patches installed over upstream mlx-lm/mlx-vlm seams. |
  | `gmlx/cache/` | It holds the automatic prompt cache and KV-cache persistence. |
  | `gmlx/spec/` | It covers speculative decoding, MTP, drafters and acceptance. |
  | `gmlx/stream/` | It covers weight streaming and residency for over-RAM models. |
  | `gmlx/serve/` | It holds the server, admission and batched decode, with HTTP patches in `serve/patches/`. |
  | `gmlx/gen/` | It holds the generation loop, sampling profiles and benchmarks. |
  | `gmlx/tune/` | It holds training-time helpers: LoRA student setup, blocked attention, per-layer checkpointing and selection ids kept off the gradient. |
  | `gmlx/distill/` | It holds offline distillation: teacher cache, cross-tokenizer alignment, fused head and losses, batch loader and eval. |
  | `gmlx/commands/` | It holds the CLI verbs behind the `gmlx` umbrella. |
  | `gmlx/tui/` | It holds the interactive chat terminal UI. |
  | `gmlx/talk/` | It holds the voice client, audio I/O and hotkey. |
  | `gmlx/assistant/` | It holds the tool-loop assistant brain and its MCP surface. |

  Cross-cutting modules such as `config.py`, `envflags.py`,
  `eval_guard.py`, `textfmt.py` and `spinner.py` stay at the `gmlx/` top
  level.
- Error messages follow the existing style: they say what was expected,
  what was found and what the user or upgrader should do next.
- Install instructions show Homebrew first, then uv, and pip only for a
  venv the user manages. The [Installation](docs/installation.md) page owns
  the install facts, and a page that adds an extra shows the uv and pip
  forms, because a Homebrew install already has every extra.

## Commit style

A commit message is a single subject line with no body, in the form
`type(scope): short lowercase summary`. The type is one of `feat`, `fix`,
`perf`, `docs`, `test` and `chore`, plus `release` for a version bump. The
scope in parentheses names the subsystem or model family the change is
about, such as `stream`, `kv`, `server`, `cli`, `distill` or `qwen4exp`, and is left
out when the change has no single home, as in `docs: fix audit findings`.
Subjects from the history include `feat(kvarn): KVarN variance-normalized
KV cache`, `fix(tokenizer): drop <|end|> from the harmony stop set for
gpt-oss` and `release: 0.4.10`. A revert is `chore(scope): revert <what>`.
