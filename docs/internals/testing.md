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

To compare CPU logits with GPU or float logits, also set `KQ_CPU_NEON=0`.
On the CPU, mlx-kquant rounds activations to int8 for a matmul of a few
rows, which moves decode logits more than the GPU's rounding does.

The docs style and link check, `scripts/check-docs.py`, is not collected
by pytest. CI runs it as a separate step, so run it yourself after editing
a doc. It also fails when a page directly under `docs/` is not linked from
the [documentation home](../README.md), so a new top-level page needs a
link there.

## Process state between tests

Each test hands the process back as it found it. `tests/process_state.py`
captures the environment, the MLX default device and streams, the memory,
cache and wired limits, the RNG key, the command buffer caps and the
Python-side copies of those settings as the first function-scoped fixture
of a test sets up, and puts them back after the last one tears down. What
a class-, module- or session-scoped fixture changes stays for the tests
that share it.

The last test of a run, `test_the_run_ends_on_production_numerics` in
`tests/test_process_state.py`, checks that the run ends on the session's
settings. It also fails when a test set a `KQ_*` key that mlx-kquant
reads once per process, since that test decided the key for the rest of
the run. Run such a test in a child process.

Set `GMLX_TEST_STATE_REPORT=1` to list, at the end of the run, each test
whose changes the restore undid and the fields it put back:

```sh
GMLX_TEST_STATE_REPORT=1 pytest tests/serve
```

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
| `tests/image/test_image_check3.py` | `GMLX_IMAGE_FIXTURES` | A Z-Image-Turbo step matches the oracle at 512x512 per block and per attention and FFN output, teacher-forced and free running, and in velocity at 512 and 1024. |
| `tests/image/test_image_load_real.py` | None | A Z-Image DiT GGUF loads with its variant from the identity hash, one `KQuantLinear` per quantized matrix and fp32 embedders, and a small step stays finite. |
| `tests/image/test_image_text_encoder_real.py` | None | A Qwen3-4B GGUF passes the Z-Image encoder checks, its tap after the last layer equals the stock forward before the final norm, and release drops the model. |
| `tests/image/test_image_check2.py` | `GMLX_IMAGE_FIXTURES` | The Z-Image encoder tap has the oracle's token ids and stays within its calibrated relative L2 and per-token cosine. |
| `tests/image/test_image_vae_real.py` | `GMLX_TEST_IMAGE_DIR` in place of `KQUANT_TEST_GGUF_DIR` | The official Z-Image VAE safetensors and a GGUF packed from it load to the same weights bit for bit. |
| `tests/image/test_image_check5.py` | `GMLX_TEST_IMAGE_DIR` and `GMLX_IMAGE_FIXTURES` in place of `KQUANT_TEST_GGUF_DIR` | On the default conv path the FLUX.1 VAE stays within the oracle's tolerances, fp32 with TF32 off in a child process, and a tiled decode stays above 35 dB. |
| `tests/serve/test_serve_apc_engagement.py` | `GMLX_TEST_BIG_GGUFS=1` adds the multi-GB rows. | One model per cache-shape family runs through the server's load path and batch engine, and its own tier counters move. |
| `tests/models/test_embedders_reference.py` | `GMLX_TEST_MLX_DIR` runs the safetensors rows. | The embeddings encoders, the EmbeddingGemma GGUF among them, match reference vectors recorded with mlx-embeddings, on single texts and on a padded batch. |

```sh
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ~/models/qwen3-0.6b
KQUANT_TEST_GGUF_DIR=~/models pytest tests/gen/test_batch_parity.py -k qwen3
```

`-k <arch>` restricts a run to one architecture, and `-m integration` runs
only the tests marked `integration`. `KQUANT_LONGCTX_TOKENS` shortens the
long-context tests for a smoke run. `KQUANT_LLAMACPP_BIN` must name
`llama-completion`, because the interactive `llama-cli` cannot run the
comparison.

`GMLX_TEST_MLX_DIR` names a folder that holds the mlx-community encoder
checkpoints the embeddings test compares, such as
`mlx-community__embeddinggemma-300m-8bit`, each under its repository name
with or without the `mlx-community__` prefix. A checkpoint that is not
there skips.

Before a release, run `tests/serve/test_serve_apc_engagement.py` with
the big rows enabled. CI has no GGUFs, so this test and the server end-to-end harnesses are the
only checks that a real model engages its cache tier.

## Image oracle fixtures

The image parity checks compare gmlx with an oracle: diffusers loading the
same GGUF file and computing on the CPU. The oracle runs in its own Python
environment, which holds torch, diffusers at the revision the scripts
name, transformers, gguf, safetensors and numpy, and never gmlx or mlx.
The oracle scripts live in `scripts/image_oracle/`, and `run.py` starts
each one in that environment as a separate process.

```sh
export GMLX_IMAGE_FIXTURES=~/image-fixtures
python scripts/image_oracle/run.py --python ~/oracle-env/bin/python \
    probe_load.py --dit z-image-turbo-Q4_K_M.gguf --config ~/Z-Image-Turbo
```

The runner refuses an interpreter that can import gmlx or mlx. It runs
the script with `HF_HUB_OFFLINE=1`, without `PYTHONPATH`, and with
`GMLX_IMAGE_FIXTURES` set, so an oracle run reads only local files.

| Name | Read by | Meaning |
|------|---------|---------|
| `GMLX_IMAGE_FIXTURES` | `scripts/image_oracle/` and the image parity tests | The directory of oracle fixtures. A test that reads a fixture skips while it is unset. |
| `GMLX_TEST_IMAGE_DIR` | The image integration tests | A directory of image model files other than text GGUFs, such as a VAE's safetensors. A test that needs one skips while it is unset. |

An oracle script writes each fixture to that directory as a `.npz` of
arrays and a `.manifest.json`. The manifest records the oracle's package
versions and revisions, the model files with their sizes and sha256, the
prompts, noise seeds and sigmas, the shape, dtype and sha256 of every
array, the oracle's own measurements and the tolerances. `manifest.json`
in the same directory indexes the fixtures. Fixtures stay outside the
repository. The repository keeps a copy of each manifest that a test
reads, under `tests/fixtures/image/oracle/`, and the test refuses a
fixture whose arrays differ from that copy. A fixture of a few tens of
KiB, such as the check 4 sigma tables, keeps its `.npz` beside the
manifest, so its test runs in the CPU tier.

The oracle runs each check in float32 and in bfloat16. A tolerance is
twice the gap between those two runs, and the manifest records the gap
beside the tolerance. The velocity also has a cosine floor of
`1 - 2(1 - c)`, where c is the oracle's own cosine between the two runs.
Cosine distance grows with the square of the angle, so the floor bounds
the angle at sqrt(2) times the oracle's own, a tighter gate than twice.
A late block adds a few percent to its output, so an error in the
block's own computation hides under the bfloat16 rounding of its output.
Each block therefore also has two branch rows, the outputs of its
attention and feed-forward modules, at twice the oracle's own gap on
each. An update row, output minus input, would inherit the residual
stream's bfloat16 rounding and hide the same error.
The oracle dequantizes GGUF blocks in float32
arithmetic, as ggml does. Stock diffusers keeps the block scales in
float16, which moves K-quant weights by about 8e-4 in relative L2, and
`probe_load.py` reports that gap for each codec. In the bfloat16 run each
linear layer multiplies the bfloat16 operands in a float32 GEMM and rounds
the result once to bfloat16, because torch's CPU bfloat16 GEMM is too slow
for a DiT. The manifest records both modes.

`check3_z_image.py --self-test` writes a tiny random Z-Image model and its
fixture. With only `GMLX_IMAGE_FIXTURES` set, `tests/image/test_image_check3.py`
checks the gmlx forward against that fixture on the CPU in float32. Its free
run takes the oracle's timestep embedding, because torch's float32 `exp`
rounds some sinusoid frequencies one ulp away from numpy's, and the random
tiny blocks amplify that gap to about 1e-5. The `t_embedder` row still
compares gmlx's own embedding with the oracle's.

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
loop. The unit tests fake audio and HTTP and cover none of the checklist.

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
