# Testing

The test suite has three tiers, ordered by what they need to run. Use the
interpreter that has gmlx and mlx-kquant installed for all of them.

| Tier | Needs | Command |
|------|-------|---------|
| CPU logic | nothing, pure Python | `pytest` |
| GGUF-gated integration | real GGUFs on disk | `KQUANT_TEST_GGUF_DIR=<dir> pytest` |
| server end-to-end | GGUFs and the GPU | the harnesses under `tests/e2e/` |

## CPU logic tests

The CPU tier runs on synthetic inputs. No model is loaded and no GPU kernel is
dispatched. It runs anywhere, including CI, and covers the remap tables,
config and tokenizer synthesis, the arch gate, weight transforms, preflight,
the config loader, the family sampling profiles, discovery, the serving id
layer, residency, the server patches and the chat client. For the chat client,
`tests/tui/test_chat_e2e.py` runs the real multi-turn loop with the model
layer faked.

```sh
pytest                       # whole suite, GGUF-gated tests skip
pytest tests/test_config.py  # one module
```

Set `KQUANT_FORCE_CPU=1` on a machine with no usable Metal GPU to keep the
few tests that use array ops off the GPU path. The doc tests under
`tests/test_docs_*.py` and `scripts/check-docs.py` are part of this tier.

## GGUF-gated integration tests

These assert numerical correctness against real weights and stay skipped until
`KQUANT_TEST_GGUF_DIR` points at a GGUF library. The directory is searched
recursively, each test selects a model by architecture from the GGUF header
and any arch you do not have skips. One small model is enough to exercise a
path.

| Module | Extra gate | What it checks |
|--------|------------|----------------|
| `tests/gen/test_batch_parity.py` | | batched decode matches single-stream |
| `tests/gen/test_long_context.py` | | long-decode integrity at 16k or more: in-range ids, finite logprobs, no single-token collapse |
| `tests/gen/test_long_context.py::test_long_prefill_parity` | `KQUANT_LLAMACPP_BIN` | long-prefill greedy output agrees with llama.cpp |
| `tests/spec/test_mtp.py`, one case | | a native-head MTP GGUF's drafter has full remap coverage |
| `tests/serve/test_serve_apc_engagement.py` | `GMLX_TEST_BIG_GGUFS=1` for the multi-GB rows | one model per cache-shape family served end to end, asserting that family's own tier counters move |

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
| `-k <arch>` | restrict to one architecture. Without it the suite sweeps each arch present |
| `KQUANT_LONGCTX_TOKENS=4096` | shrink the long-context length from the 16384 default |
| `KQUANT_LLAMACPP_BIN=/path/to/llama-completion` | enable the llama.cpp parity tests. `llama-cli` is interactive-only on new builds and hangs the helper |
| `-m integration` | only the marker-carrying parity modules |

Before a release, run the engagement gate with the big rows enabled. CI has
no GGUFs, which makes this the one check that proves a real served model
engages its cache tier:

```sh
KQUANT_TEST_GGUF_DIR=~/llm/gguf-test GMLX_TEST_BIG_GGUFS=1 \
  pytest tests/serve/test_serve_apc_engagement.py -v
```

## Server end-to-end harnesses

`tests/e2e/` holds standalone scripts that launch the real server, load models
on the GPU and grade the results. They are not part of the pytest suite,
though `tests/test_e2e_harness_smoke.py` checks every harness's imports and
argument tree in CI. Each harness is described, with its tiers, grading and
model bootstrap, in [tests/e2e/README.md](../../tests/e2e/README.md).

| Harness | Exercises |
|---------|-----------|
| `run_server_e2e.py` | the start-mode and config matrix with a graded prompt suite |
| `run_capacity_e2e.py`, `run_capacity_soak_e2e.py`, `run_capacity_multi_e2e.py` | metrics, queue and governor invariants under load, single and multi model |
| `run_residency_switch_e2e.py` | two models that cannot both be resident |
| `run_stream_e2e.py` | a streamed model through load cycles, memory pressure and coresidency, with `memguard.py` run beside it |
| `run_apc_disk_e2e.py`, `run_apc_depth_e2e.py` | prompt-cache reuse across restarts and at depth, per tier |
| `run_lora_e2e.py` | prep, train, serve base and adapter, assert the adapter changed the output style |
| `run_chat_pty_e2e.py` | the chat client in a real pseudo-terminal |

```sh
python tests/e2e/run_server_e2e.py --print-pull   # pull commands for the harness models
python tests/e2e/run_server_e2e.py --dry-run      # CPU-only: validate the config matrix
python tests/e2e/run_server_e2e.py                # full run, writes report.md and report.json
```

## Voice loop manual pass

A manual checklist after changing the `gmlx talk` loop. The unit tests fake
audio and HTTP and cover none of this.

1. With the server down, `gmlx talk` autostarts it, the capability check
   passes and the prompt appears.
2. A wake phrase, a question and a spoken reply. Time end-of-speech to first
   audio.
3. Space mid-reply stops speech quickly. The next wake still works.
4. `/voice` switches to a Kokoro preset and, if configured, a qwen3-tts
   speaker.
5. A long multi-sentence answer plays without gaps or underruns.
6. 60 s of silence and 60 s of background noise produce no ghost turns.
7. `--once` exits after one exchange. `--mode text` speaks the replies to typed input.
8. The menu bar "Talk to model" item opens a working session.
