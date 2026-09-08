# Glossary

Terms used throughout these docs and on Hugging Face model pages, ordered
roughly as you will meet them.

- **Open-weight model**: a language model whose weights are published for
  anyone to download and run - Llama, Qwen, Gemma, DeepSeek, GLM, gpt-oss,
  and many more. Running one locally keeps the conversation on your Mac.
- **GGUF**: the single-file format the open-model ecosystem publishes models
  in. One `.gguf` file is one ready-to-run model (very large ones are split
  into numbered shards; gmlx treats the set as one file). gmlx runs GGUFs
  exactly as published, with no conversion step.
- **Quant (quantization)**: a compressed build of a model. The suffix on a
  GGUF name - `Q4_K_M`, `Q6_K`, `Q8_0`, `IQ2_M` - says how many bits each
  weight keeps: the number is roughly bits per weight, so Q4 files are
  smaller and slightly lossier, Q6/Q8 bigger and closer to the original.
  Most repos publish one file per quant; you pick the size that fits your
  RAM ([pick a model](getting-started.md#pick-a-model-for-your-mac)).
- **Token**: the unit models read and write - a word fragment, on average
  about three-quarters of an English word. Speeds are quoted in tokens per
  second (tok/s).
- **Context**: everything the model is currently considering - your
  conversation so far, pasted files, its own reply in progress - measured in
  tokens.
- **KV cache**: the model's working memory for the context, kept alongside
  the weights in RAM. It grows with context length, which is why a model
  whose file barely fits leaves no room for long conversations
  ([will it fit?](getting-started.md#will-it-fit)); `--kv-bits 8` compresses it.
- **Prefill and decode**: the two phases of answering. Prefill is reading
  your prompt (fast, reported separately); decode is generating the reply
  token by token. `gmlx run`'s closing stats line reports both speeds.
- **Thinking (reasoning) model**: a model trained to reason step by step
  before answering, streaming that scratch work inside `<think>` markers.
  `run` and `chat` style it under a `thinking` label by default
  (`--reasoning` controls this).
- **MoE (mixture of experts)**: a model built from many small "expert"
  sub-networks, of which each token activates only a few, so decode costs
  what the active fraction costs rather than the full parameter count (the
  `-A3B` in a name like `35B-A3B` = 3B active parameters). Because most
  experts are idle on any given token, a MoE bigger than RAM can still run by
  [streaming experts from disk](streaming.md).
- **Speculative decoding / MTP**: a small draft predictor proposes several
  tokens and the full model verifies them in one step - identical output,
  fewer full-model passes. Some models (qwen3.5/3.6)
  ship the draft head inside the GGUF ("MTP", multi-token prediction), and
  gmlx turns it on automatically.
- **mmproj**: a companion GGUF holding a vision (or audio) tower. Pair it
  with its LLM GGUF via `--mmproj` and the model can see images
  ([vlm.md](vlm.md)).
- **Hugging Face**: the site the open-model ecosystem publishes on.
  `hf:org/repo/file.gguf` refs throughout these docs point there; `gmlx
  pull` downloads them.
- **OpenAI-compatible API**: the de-facto standard HTTP interface for
  chatting with a model. `gmlx serve` speaks it, and Anthropic's, so clients
  written for those APIs address a local server unchanged.
