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
subsystem, from the loader's remap tables and config synthesis through the
config loader, discovery, residency and the server patches to the chat
client. For the client, `tests/tui/test_chat_e2e.py` drives the real
multi-turn loop with the model layer faked. The doc tests under
`tests/test_docs_*.py` belong to this tier as well.

```sh
pytest                       # whole suite, GGUF-gated tests skip
pytest tests/test_config.py  # one module
```

Set `KQUANT_FORCE_CPU=1` on a machine with no usable Metal GPU to keep the
few tests that use array ops off the GPU path.

The docs style and link check, `scripts/check-docs.py`, is not collected
by pytest. CI runs it as a separate step, so run it yourself after editing
a doc. It also fails when a page in `docs/` is not linked from
the [documentation home](../README.md) and from the Documentation list in the
[project README](../../README.md#documentation), so a new page needs both
links.

## GGUF-gated integration tests

These assert numerical correctness against real weights and stay skipped
until `KQUANT_TEST_GGUF_DIR` points at a GGUF library. The directory is
searched recursively, each test selects a model by architecture from the
GGUF header, and any arch you do not have skips, so one small model is
enough to exercise a path.

| Module | Extra gate | What it checks |
|--------|------------|----------------|
| `tests/gen/test_batch_parity.py` | None. | Batched decode matches single-stream decode. |
| `tests/gen/test_long_context.py` | None. | Long decodes of 16k tokens or more keep ids in range, logprobs finite and output free of single-token collapse. |
| `tests/gen/test_long_context.py::test_long_prefill_parity` | `KQUANT_LLAMACPP_BIN` | Long-prefill greedy output agrees with llama.cpp. |
| `tests/spec/test_mtp.py`, one case | None. | A native-head MTP GGUF's drafter has full remap coverage. |
| `tests/serve/test_serve_apc_engagement.py` | `GMLX_TEST_BIG_GGUFS=1` enables the multi-GB rows. | One model per cache-shape family is served end to end, and that family's own tier counters must move. |

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ~/models/qwen3-0.6b
KQUANT_TEST_GGUF_DIR=~/models pytest tests/gen/test_batch_parity.py -k qwen3

# the long-context layer on a small sliding-window model, shortened for a smoke
gmlx pull hf:ggml-org/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf --to ~/models/gemma-3-1b-it-GGUF
KQUANT_TEST_GGUF_DIR=~/models KQUANT_LONGCTX_TOKENS=4096 \
  pytest tests/gen/test_long_context.py::test_long_decode_integrity -k gemma3
```

| Setting | Effect |
|------|--------|
| `-k <arch>` | Restricts the run to one architecture. Without it the suite sweeps each arch present. |
| `KQUANT_LONGCTX_TOKENS=4096` | Shrinks the long-context length from the 16384 default. |
| `KQUANT_LLAMACPP_BIN=/path/to/llama-completion` | Enables the llama.cpp parity tests. An interactive-only `llama-cli` fails the run with a message naming `llama-completion`. |
| `-m integration` | Runs only the parity modules that carry the marker. |

Before a release, run the engagement gate with the big rows enabled. CI has
no GGUFs, which makes this the one check that proves a real served model
engages its cache tier.

```sh
KQUANT_TEST_GGUF_DIR=~/models GMLX_TEST_BIG_GGUFS=1 \
  pytest tests/serve/test_serve_apc_engagement.py -v
```

A serve performance claim is likewise measured in the real server process,
with the round profile switches listed in [Debug
switches](debug-switches.md).

## Server end-to-end harnesses

`tests/e2e/` holds standalone scripts that launch the real server, load
models on the GPU and grade the results. They are not part of the pytest
suite, although `tests/test_e2e_harness_smoke.py` checks every harness's
imports and argument tree in CI. Each harness is described, with its tiers,
grading and model bootstrap, in [Server end-to-end test
harness](../../tests/e2e/README.md).

| Harness | What it exercises |
|---------|-----------|
| `run_server_e2e.py` | Every start mode and config in the matrix runs a graded prompt suite. |
| `run_capacity_e2e.py`, `run_capacity_soak_e2e.py`, `run_capacity_multi_e2e.py` | Metrics, queue and governor invariants are checked under load, with one model and with several. |
| `run_residency_switch_e2e.py` | The server switches between two models that cannot both be resident. |
| `run_stream_e2e.py` | A streamed model goes through load cycles, memory pressure and coresidency, with `memguard.py` run beside it. |
| `run_apc_disk_e2e.py`, `run_apc_depth_e2e.py` | Prompt-cache reuse is checked across restarts and at depth, for each tier. |
| `run_lora_e2e.py` | The harness preps, trains and serves base and adapter, then checks that the adapter changed the output style. |
| `run_distill_e2e.py` | The harness caches a small teacher, aligns, trains an adapter and evals before and after, then checks that the loss fell. |
| `run_chat_pty_e2e.py` | The chat client runs in a real pseudo-terminal. |
| `run_serve_harmony_e2e.py` | A served gpt-oss model keeps its response contract, with no harmony channel markup in content and truncation inside analysis. |
| `run_serve_stress_e2e.py` | Seeded concurrent chaos hits one server, with mid-stream aborts, tiny budgets, warm resends and growing sessions. |
| `run_systemone_e2e.py` | `/v1/systemone` runs on a DiffusionGemma GGUF, covering answer shapes, obvious answers, seed replay, refusals, a concurrent chat and `gmlx systemone`. |

```sh
python tests/e2e/run_server_e2e.py --print-pull   # pull commands for the harness models
python tests/e2e/run_server_e2e.py --dry-run      # CPU-only: validate the config matrix
python tests/e2e/run_server_e2e.py                # full run, writes report.md and report.json
```

## Voice loop manual pass

Run this checklist by hand before merging a change to the `gmlx talk`
loop. The unit tests fake audio and HTTP and cover none of it.

1. With the server down, `gmlx talk` autostarts it, the capability check
   passes and the prompt appears.
2. A question after the wake phrase gets a spoken reply. Time
   end-of-speech to first audio.
3. Space mid-reply stops speech quickly. The next wake still works.
4. `/voice` switches to a Kokoro preset and, if configured, a qwen3-tts
   speaker.
5. A long multi-sentence answer plays without gaps or underruns.
6. 60 s of silence and 60 s of background noise produce no ghost turns.
7. `--once` exits after one exchange. `--mode text` speaks the replies to
   typed input.
8. The menu bar "Talk to model" item opens a working session.
