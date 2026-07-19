# Testing

The test suite has three layers, ordered by what they need to run:

| layer | needs | command |
| --- | --- | --- |
| **CPU logic** (default) | nothing, pure Python | `pytest` |
| **GGUF-gated integration** | real GGUF(s) on disk | `KQUANT_TEST_GGUF_DIR=<dir> pytest` |
| **Server end-to-end** | GGUF(s) + GPU | `python tests/e2e/run_server_e2e.py` |

Use the interpreter that has `gmlx` + `mlx_kquant` installed for all of them.

## 1. CPU logic tests (no models)

The CPU tier runs everything on synthetic inputs: no model is loaded and no GPU
kernel is dispatched, so it runs anywhere, including CI. It covers the remap
tables, config/tokenizer synthesis, the arch gate, weight transforms, preflight,
the config loader, the family sampling profiles, discovery, the serving id-layer,
residency, the server patches, and the `chat` REPL, where `tests/test_chat_e2e.py`
drives the real multi-turn loop with the model layer faked, plus the live
prompt_toolkit session over a pipe:

```sh
pytest                       # whole suite; GGUF-gated tests auto-skip (see below)
pytest tests/test_config.py  # one module
```

Set `KQUANT_FORCE_CPU=1` on a box with no usable Metal GPU to keep the few tests that
touch `mx` array ops off the GPU path.

## 2. GGUF-gated integration tests (need a model)

Three modules assert numerical correctness against real weights and stay skipped until
you point the suite at a GGUF library:

| module | gate | what it checks |
| --- | --- | --- |
| `tests/test_batch_parity.py` | `KQUANT_TEST_GGUF_DIR` | batched decode is faithful to single-stream (b=1 token-exact, uniform-batch determinism, ragged divergence only at logit ties) |
| `tests/test_long_context.py` | `KQUANT_TEST_GGUF_DIR` | long-decode integrity at >=16k (in-range, finite logprobs, no single-token collapse); attention bugs only surface at depth |
| `tests/test_mtp.py` (one case) | `KQUANT_TEST_GGUF_DIR` | a native-head MTP GGUF's drafter has full remap coverage (the rest of the module is CPU-only) |
| `tests/test_long_context.py::test_long_prefill_parity` | also `KQUANT_LLAMACPP_BIN` | long-prefill greedy output agrees with llama.cpp |

`KQUANT_TEST_GGUF_DIR` is searched recursively for `*.gguf`. Each test selects a model
by architecture (read from the GGUF header), so it auto-skips any arch you don't have.
You don't need a full zoo: one small model is enough to exercise a path.
The env var alone enables these tests; `KQUANT_TEST_GGUF_DIR=<dir> pytest` is the
canonical invocation. Adding `-m integration` restricts the run to the
marker-carrying parity modules.

### One-liner: get a model, then run

```sh
# fetch a tiny public model (header-checked first, then downloaded, not into the HF cache)
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ~/models/qwen3-0.6b

# run the batched-decode parity suite against just that arch
KQUANT_TEST_GGUF_DIR=~/models pytest tests/test_batch_parity.py -k qwen3
```

For the long-context layer, a small SWA model exercises it (Qwen3 isn't in its arch sweep):

```sh
gmlx pull hf:ggml-org/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf --to ~/models/gemma-3-1b-it-GGUF

# quick pass: shorten the 16k default so a small model finishes in seconds
KQUANT_TEST_GGUF_DIR=~/models KQUANT_LONGCTX_TOKENS=4096 \
  pytest tests/test_long_context.py::test_long_decode_integrity -k gemma3
```

Knobs:

- `-k <arch>`: restrict to one architecture (`qwen2`, `qwen3`, `gemma3`, `gemma4`,
  `qwen35moe`, `llama`, ...). Without it the suite sweeps every arch present and can take
  minutes on large models.
- `KQUANT_LONGCTX_TOKENS=4096`: shrink the long-context length for a fast smoke (default
  16384, capped per model's context window).
- `KQUANT_LLAMACPP_BIN=/path/to/llama-completion`: enables the llama.cpp parity tests
  (`test_long_prefill_parity`). Without it they skip and only the self-contained integrity
  checks run. On newer llama.cpp builds the binary is `llama-completion`; `llama-cli`
  there is interactive-only and will hang the parity helper.

Point `KQUANT_TEST_GGUF_DIR` at a whole library (e.g. `~/models`) to sweep every arch you
have in one run; drop `-k` to do so.

## 3. Server end-to-end harness

`tests/e2e/` launches the real `gmlx.server` across a matrix of start modes and
config features, fires a prompt suite at each live server, and grades every response (floor
checks + an LLM-as-judge). It loads models and needs the GPU, so it is not part of the
pytest suite. Its own guide (tiers, grading, model bootstrap, output format) lives in
[`tests/e2e/README.md`](../tests/e2e/README.md). Quick start:

```sh
# no models on disk yet? print copy-paste pull commands for the harness's models
python tests/e2e/run_server_e2e.py --print-pull

# CPU-only: build + validate the whole config matrix (loads no model); run this first
python tests/e2e/run_server_e2e.py --dry-run

# full run (GPU); writes report.md + report.json
python tests/e2e/run_server_e2e.py
```

`tests/e2e/run_apc_disk_e2e.py` exercises disk-backed APC prefix reuse across
server restarts the same way: real server, real GGUF, standalone.

### LoRA-on-GGUF end-to-end

`tests/e2e/run_lora_e2e.py` exercises the whole GGUF LoRA loop as a user runs it, from
creation through serving: prep a tiny finetune set, `gmlx train` a LoRA on a small
K-quant GGUF base (emitting a GGUF adapter), then `gmlx serve --adapter` the base and
assert the served output shifted. The pirate finetune is graded by a deterministic marker
check with greedy decoding. Both verbs run as real subprocesses; it needs a base GGUF + the
GPU, so it's standalone, not pytest. The adapter writer + train-driver fidelity are
CPU-tested in `tests/test_adapter_save.py` / `tests/test_train.py`.

```sh
python tests/e2e/run_lora_e2e.py                     # prep -> train -> serve -> assert
python tests/e2e/run_lora_e2e.py --reuse-adapter A.gguf   # skip training, serve only
```

### Chat TUI (pty) end-to-end

`tests/e2e/run_chat_pty_e2e.py` drives the interactive `gmlx chat` UI in a real
pseudo-terminal (stdlib `pty`, no `pexpect`). It is the only tier that exercises
the tty-only surface: the live prompt_toolkit session, the streaming reply, and
the termios Esc-cancel. It walks a scripted session: load + banner, then two turns over one
KV cache, then a live `/temp` + `/sampling` slash command, then Esc-cancel of a long reply,
then `q` quits clean. An optional multimodal arm (stage `assets/cats.jpg` with `/image`,
generate about it) runs when a VLM GGUF + projector are on disk. It loads a model and needs the GPU, so it's standalone, not
pytest. A missing model is a SKIP. The deterministic loop + session coverage is CPU-tested,
no tty, in `tests/test_chat_e2e.py`.

```sh
python tests/e2e/run_chat_pty_e2e.py                 # text arm (+ vlm if present)
python tests/e2e/run_chat_pty_e2e.py --no-vlm        # text arm only
python tests/e2e/run_chat_pty_e2e.py --print-pull    # how to fetch the model
```

### Voice loop manual pass

A manual checklist after touching the `gmlx talk` loop. Nothing here is covered by
unit tests, which fake audio and HTTP:

1. Server down, `gmlx talk`: autostarts, capability check passes, prompt appears.
2. Wake phrase, question, spoken reply; stopwatch end-of-speech to first audio.
3. Space mid-reply stops speech quickly; the next wake still works.
4. `/voice` switch (a Kokoro preset and, if configured, a qwen3-tts speaker).
5. A long multi-sentence answer plays without gaps or underruns.
6. 60 s of silence and of background noise: no ghost turns.
7. `--once` exits after one exchange; `--mode text` speaks typed input's replies.
8. The menu bar "Talk to model" item opens a working session.
