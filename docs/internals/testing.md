# Testing

The test suite has three tiers, ordered by what they need to run. Use the
interpreter that has gmlx and mlx-kquant installed for all of them.

| Tier | Needs | Command |
|------|-------|---------|
| CPU logic | It needs nothing beyond Python. | `pytest` |
| GGUF-gated integration | It needs real GGUFs on disk. | `KQUANT_TEST_GGUF_DIR=<dir> pytest` |
| Server end-to-end | It needs GGUFs and the GPU. | Run the harnesses under `tests/e2e/`. |

## CPU logic tests

The CPU tier runs on synthetic inputs, with no model loaded and no GPU
kernel dispatched, so it runs anywhere, including CI. It reaches every
subsystem, from the loader's remap tables and config synthesis to the
server patches and the chat client. The doc tests under
`tests/test_docs_*.py` belong to this tier as well.

```sh
pytest                       # whole suite, GGUF-gated tests skip
pytest tests/test_config.py  # one module
```

Set `KQUANT_FORCE_CPU=1` on a machine with no usable Metal GPU to keep the
few tests that use array ops off the GPU path.

The docs style and link check, `scripts/check-docs.py`, is not collected
by pytest. CI runs it as a separate step, so run it yourself after editing
a doc. It also fails when a page directly under `docs/` is not linked from
the [documentation home](../README.md) and from the Documentation list in
the [project README](../../README.md#documentation), so a new top-level
page needs both links.

## GGUF-gated integration tests

The GGUF-gated tests check numerical correctness against real weights and
stay skipped until `KQUANT_TEST_GGUF_DIR` points at a folder of GGUFs.
Each test selects a model by architecture from the GGUF header, and any
architecture you do not have skips, so one small model is enough to
exercise a path.

| Module | Extra gate | What it checks |
|--------|------------|----------------|
| `tests/gen/test_batch_parity.py` | None | A batch of one matches single-stream decode, and a ragged batch diverges from it only at exact logit ties. |
| `tests/gen/test_long_context.py` | None | A long decode keeps its ids in range, its logprobs finite and its output free of single-token collapse. |
| `tests/gen/test_long_context.py::test_long_prefill_parity` | `KQUANT_LLAMACPP_BIN` | Long-prefill greedy output agrees with llama.cpp. |
| `tests/spec/test_full_prompt_prefill.py` | `KQUANT_TEST_MTP_GGUF` | The MTP serve path handles prefill, the prompt cache, batching and injection. |
| `tests/serve/test_serve_apc_engagement.py` | `GMLX_TEST_BIG_GGUFS=1` adds the multi-GB rows. | One model per cache-shape family runs through the server's load path and batch engine, and its own tier counters move. |

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ~/models/qwen3-0.6b
KQUANT_TEST_GGUF_DIR=~/models pytest tests/gen/test_batch_parity.py -k qwen3
```

`-k <arch>` restricts a run to one architecture, and `-m integration` runs
only the tests marked `integration`. `KQUANT_LONGCTX_TOKENS` shortens the
long-context tests for a smoke run. `KQUANT_LLAMACPP_BIN` must name
`llama-completion`, because the interactive `llama-cli` cannot run the
comparison.

Before a release, run the engagement gate with the big rows enabled. CI
has no GGUFs, so this gate and the server end-to-end harnesses are the
only checks that a real model engages its cache tier.

## Server end-to-end harnesses

`tests/e2e/` holds standalone scripts that launch the real server, load
models on the GPU and grade the results. They are not part of the pytest
suite, although `tests/test_e2e_harness_smoke.py` checks every harness's
imports and arguments in CI.
[Server end-to-end test harness](../../tests/e2e/README.md) describes each
harness, its tiers, its grading and its model bootstrap.

```sh
python tests/e2e/run_server_e2e.py --print-pull   # pull commands for the harness models
python tests/e2e/run_server_e2e.py --dry-run      # CPU-only: validate the config matrix
python tests/e2e/run_server_e2e.py                # full run, writes report.md and report.json
```

A serve performance claim is measured in the real server process, with
the round profile switch that [Debug switches](debug-switches.md#profiling)
lists.

## Voice loop manual pass

Run this checklist by hand before merging a change to the `gmlx talk`
loop. The unit tests fake audio and HTTP and cover none of it.

1. With the server down, `gmlx talk` starts it, and the prompt appears.
2. A question after the wake phrase gets a spoken reply.
3. Space in the middle of a reply stops speech quickly, and the next wake
   still works.
4. `/voice` switches to another voice.
5. A long answer of many sentences plays without gaps.
6. A minute of silence and a minute of background noise produce no ghost
   turns.
7. `--once` exits after one exchange, and `--mode text` speaks the replies
   to typed input.
8. The menu bar item that talks to a model opens a working session.
