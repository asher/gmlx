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
| `tests/image/test_image_check3_qwen.py` | `GMLX_IMAGE_FIXTURES` | A Qwen-Image 2.1 step matches the oracle per block at 512x512 and in velocity at 1024, and each op of a block its stock op. |
| `tests/image/test_image_qwen_real.py` | None | A Qwen-Image 2.1 DiT GGUF loads with fp32 norm leaves, and a small prefill and cached step stays finite with and without a reference. |
| `tests/image/test_image_load_real.py` | None | A Z-Image DiT GGUF loads with its variant from the identity hash, one `KQuantLinear` per quantized matrix and fp32 embedders, and a small step stays finite. |
| `tests/image/test_image_text_encoder_real.py` | None | A Qwen3-4B GGUF passes the Z-Image encoder checks, its last-layer tap equals the copied mlx-lm forward before the final norm, and release drops the model. |
| `tests/image/test_image_check2.py` | `GMLX_IMAGE_FIXTURES` | The Z-Image encoder tap has the oracle's token ids and stays within its calibrated relative L2 and per-token cosine. |
| `tests/image/test_image_vae_real.py` | `GMLX_TEST_IMAGE_DIR` in place of `KQUANT_TEST_GGUF_DIR` | The official Z-Image VAE safetensors and a GGUF packed from it load to the same weights bit for bit. |
| `tests/image/test_image_check5.py` | `GMLX_TEST_IMAGE_DIR` and `GMLX_IMAGE_FIXTURES` in place of `KQUANT_TEST_GGUF_DIR` | On the default conv path the FLUX.1 VAE stays within the oracle's tolerances, fp32 with TF32 off in a child process, tiled passes included. |
| `tests/image/test_image_qwen_vae_real.py` | `GMLX_TEST_IMAGE_DIR` in place of `KQUANT_TEST_GGUF_DIR` | The official Qwen-Image 2.1 VAE safetensors has the committed header and sha256, and a GGUF packed from it keeps F32 and loads to the same weights bit for bit. |
| `tests/image/test_image_check5_qwen.py` | `GMLX_TEST_IMAGE_DIR`, `GMLX_IMAGE_FIXTURES` and a committed `check5_qwen_image21_vae` manifest, in place of `KQUANT_TEST_GGUF_DIR` | The official Qwen-Image 2.1 VAE holds the oracle's tolerances in guarded passes, whole up to 1024 pixels and in tiles. Its tiny tests run in the CPU tier. |
| `tests/image/test_image_cli_real.py` | `GMLX_TEST_IMAGE_DIR` | `gmlx image generate` writes the latents that `sample` returns for the noise of the same seed, bit for bit. |
| `tests/image/test_image_check6.py` | `GMLX_TEST_IMAGE_DIR` and `GMLX_IMAGE_FIXTURES` | A whole Z-Image-Turbo run from the oracle's noise stays within the PSNR gate of the oracle's fp32 image. |
| `tests/image/test_image_check7.py` | `GMLX_IMAGE_FIXTURES` | A Z-Image-Turbo step at 512x512 under a live LoRA matches the oracle on merged weights block by block and in velocity. The base and each wrong table fail. |
| `tests/image/test_image_check7_community.py` | `GMLX_TEST_LORA_DIR` and `GMLX_IMAGE_FIXTURES` | Public Z-Image-Turbo LoRA files load with every key matched, give the float64 delta, and pass the step against the merged oracle block by block. |
| `tests/image/test_image_check6_qwen.py` | `GMLX_IMAGE_FIXTURES` | A whole Qwen-Image 2.1 run from the oracle's conditioning and noise ends within the gate of the oracle's fp32 latents, on its schedule. |
| `tests/image/test_image_check7_qwen.py` | `GMLX_IMAGE_FIXTURES` | A Qwen-Image 2.1 step at 512x512 under one live LoRA or two matches the oracle on merged weights, and each leaf its exact product. Wrong tables fail. |
| `tests/serve/test_serve_apc_engagement.py` | `GMLX_TEST_BIG_GGUFS=1` adds the multi-GB rows. | One model per cache-shape family runs through the server's load path and batch engine, and its own tier counters move. |
| `tests/models/test_qwen3vl_integration.py` | None | A Qwen3-VL pair loads with every tensor used, the text model and the pair agree before the final norm, and prompt layout and image features match llama.cpp. |
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
| `GMLX_TEST_LORA_DIR` | `tests/image/test_image_check7_community.py` | A directory of the public LoRA files the check 7 manifest names, found by file name and checked by sha256. The tests skip while it is unset. |

An oracle script writes each fixture to that directory as a `.npz` of
arrays and a `.manifest.json`. The manifest records the oracle's package
versions and revisions, the model files with their sizes and sha256, the
prompts, noise seeds and sigmas, the shape, dtype and sha256 of every
array, the oracle's own measurements and the tolerances. `manifest.json`
in the same directory indexes the fixtures. Fixtures stay outside the
repository. The repository keeps a copy of each manifest that a test
reads, under `tests/fixtures/image/oracle/`, and the test refuses a
fixture whose arrays differ from that copy. A fixture of at most a few
hundred KiB, such as the check 4 sigma tables and the tiny VAEs of check
5 for Qwen-Image 2.1, keeps its `.npz` beside the manifest, so its test
runs in the CPU tier.

The oracle runs each check in float32 and in bfloat16. A tolerance is
twice the gap between those two runs, and the manifest records the gap
beside the tolerance. The velocity also has a cosine floor of
`1 - 4(1 - c)`, where c is the oracle's own cosine between the two runs.
Cosine distance grows with the square of the angle, so the floor bounds
the angle at twice the oracle's own, as the tolerance bounds the
relative L2.
A late block adds a few percent to its output, so an error in the
block's own computation hides under the bfloat16 rounding of its output.
Each block therefore also has two branch rows, the outputs of its
attention and feed-forward modules. An update row, output minus input,
would inherit the residual stream's bfloat16 rounding and hide the same
error.

Checks 3 and 7 read one denoise step through three gates, which
`tests/image/block_gates.py` holds for both. The exact gate proves the
structure: the model in the oracle's float32 arithmetic, with exact
float32 products, every K-quant product on its dequantized weight and the
stock ops, matches the oracle on every row and branch row at `5e-5`, and
each listed wrong model or wrong adapter table fails it. Check 7 also
runs it with `GMLX_IMAGE_LORA_PLACEMENT=split`, where a K-quant product
runs once per output band past its module's call. The product gate
proves the result of the path a request takes: the block rows, the
free-running blocks and the velocity are within the manifest's
tolerances, each twice the oracle's own bfloat16 against float32 gap at
that point. The site gate proves the arithmetic of each op on that path:
the output of every op between a block's input and its output is within
a relative L2 of `2 n u` of the stock op in float32 on the same input,
where `u = 2^-8` is one bfloat16 rounding and `n` counts the outputs the
stock op writes in bfloat16, so a kernel is held to the gate of the op it
replaces. MLX reads `MLX_ENABLE_TF32` once per process, so the exact and
the site tests rerun themselves in a child process with
`MLX_ENABLE_TF32=0`, the exact tests with the image ops on the stock path
and the site tests on the kernel routes the build has.
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

`check3_qwen_image21.py --self-test` writes a tiny Qwen-Image 2.1 model
in GGUF tensor names, loads it back through diffusers' single-file path
and runs one conditioning through three calls: the joint forward of the
whole sequence, the extract call that keeps each block's K and V of the
prefix, and the cached decode of the target rows at another sigma. It
writes four fixtures small enough to commit beside their manifests: a
text prefix, one reference image, two reference images with every block
row, and the token layouts of five conditionings.
`scripts/image_oracle/qwen_image21_tiny.py` forms the tiny model's
weights and inputs from integer arithmetic, so the oracle and the gmlx
tests build the same arrays with no weights file between them, and each
manifest holds the sha256 of the weights.
`tests/image/test_image_check3_qwen.py` loads the same weights from a
GGUF and runs the gmlx prefill and target forward against these fixtures
on the CPU in float32, with the rows that `tests/image/qwen_gates.py`
describes. Every row is held to `4e-6`, about seven times the largest
reading, and each wrong model of its list must miss a row by twice that.
`tests/image/test_image_qwen_layout.py` holds the position of every
token, the expansion of image slots, the modulation row of every token
and the prefill segments to the layout fixture. The rows of
`test_image_check3_qwen.py` that read the real file skip until the
manifests that `check3_qwen_image21.py` writes from it are committed.

`check5_qwen_image21_vae.py --self-test` builds diffusers'
`AutoencoderKLQwenImage21` at a tiny config, with the seeded weights of
`qwen_image21_vae_tiny.py`, and writes a fixture that the repository
keeps. `tests/image/test_image_check5_qwen.py` writes the same weights as
a safetensors file, whose sha256 the manifest holds, packs it into a GGUF,
and checks the gmlx decode and encode from both files against the fixture
on the CPU. Nine wrong models, each one departure from the reference, must
fail the float32 gate. Without `--self-test` the script runs on the
official VAE and writes the fixture of the gated tests.

`check7_qwen_image21_lora.py --self-test` merges the adapters of
`qwen_image21_tiny.py` into the dense weights of that tiny model and runs
the same three calls. It writes one fixture per variant: an adapter with
an alpha, one without, and a stacked pair with a negative strength. The
first manifest also records, for every linear layer of the reference
class and every key spelling of an adapter file, the layer that
diffusers' own Qwen LoRA loader gives the key.
`tests/image/test_image_check7_qwen.py` writes the same pairs in each
dialect and runs the step under a patch table, on a model that holds the
fused gate and projection weight and on one that holds the pair apart.
Every row is held to the bounds of check 3, and each wrong table of its
list must miss a bound by twice. Its rows that read the real file skip
until the manifests that `check7_qwen_image21_lora.py` writes from it
are committed. On the real file the script seeds the pairs itself, at a
norm of 0.15 times each weight, and writes an adapter with an alpha, one
without, and a stack of two on different layers and blocks.

`check2_qwen_image21.py` loads Qwen3-VL-8B from its text GGUF into the
transformers class by tensor name and runs the pipeline's own prompt
encoding. `check6_qwen_image21.py` takes one of its taps as the
conditioning and runs the pipeline's own call on injected noise, with
the bf16 run as a loop of the same calls. Both manifests are committed,
and `tests/image/test_image_check6_qwen.py` reads the second.

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
