# Byte-level pre-tokenizers

gmlx builds a GGUF's tokenizer from the GGUF header. For a byte-level BPE
vocab, the header key `tokenizer.ggml.pre` names how text splits into words
before the merges run. gmlx follows llama.cpp for each name, so a prompt
gets the same token ids in both.

## The table

`PRE_TOKENIZERS` in `gmlx/load/tokenizer.py` maps each name to an entry with
four columns:

| Column | Meaning |
|--------|---------|
| `patterns` | The split regexes, applied in sequence. Each regex splits every piece that the one before it made. |
| `ignore_merges` | A word that is a whole vocab entry becomes one id, and the merges do not run on it. |
| `add_bos` | The BOS default for a GGUF without `tokenizer.ggml.add_bos_token`. |
| `nfc` | Apply the NFC normalizer before the split. |

The regexes are llama.cpp's. Where llama.cpp keeps the regex of the original
`tokenizer.json` in a comment, the table uses that spelling. The tokenizers
library runs the regexes on its own engine, and a test checks that the
pieces match the pieces the `regex` module gives.

Names that share a regex set share one entry, as in llama.cpp. The table
also keeps three names that llama.cpp does not list, for GGUFs that already
carry them: `qwen3`, `qwen35moe` and `hyv4`.

## The BOS default

llama.cpp turns BOS on for the Llama 3 and Tekken groups. The GGUF key
`tokenizer.ggml.add_bos_token` overrides that whenever the GGUF has it. gmlx
does the same: a missing key takes the entry's `add_bos`, and a present key
wins. Every other entry leaves a missing key as off.

The table does not model llama.cpp's `null_bos` flag, which `chatglm-bpe`
sets.

## Unknown names

A name that the table does not list, or an empty name, falls back to the
Qwen 2 split with no NFC. The load prints one warning that names the
pre-tokenizer, because token ids can then differ from llama.cpp. An SPM vocab, such as Gemma's, has no word split
and gets no warning.

## Unicode normalization

llama.cpp applies no normalization to a BPE vocab. The `nfc` column is off
by default, so an entry matches llama.cpp unless it sets the column. An entry
sets it only where the reference `tokenizer.json` declares an NFC
normalizer: `qwen2`, `qwen3`, `qwen35`, `qwen35moe`, `deepseek-r1-qwen` and
`k2-horizon`. These entries match Hugging Face, and differ from llama.cpp on
text that is not in NFC, such as an `e` followed by a combining accent.
Every other entry matches llama.cpp. Each of those whose reference file was
read declares no normalizer, so it matches Hugging Face too.

The normalizer field was read from the reference file of these entries:

| Entry | Reference repository | Normalizer |
|-------|----------------------|------------|
| `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct` | NFC |
| `qwen3` | `Qwen/Qwen3-0.6B` | NFC |
| `qwen35` | `Qwen/Qwen3.5-9B` | NFC |
| `qwen35moe` | `Qwen/Qwen3.5-35B-A3B` | NFC |
| `deepseek-r1-qwen` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | NFC |
| `k2-horizon` | `IFM/K2-Horizon-7B` | NFC |
| `llama-bpe` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | none |
| `falcon3` | `tiiuae/Falcon3-1B-Instruct` | none |
| `falcon-h1` | `tiiuae/Falcon-H1-0.5B-Instruct` | none |
| `lfm2` | `LiquidAI/LFM2-350M` | none |
| `glm4`, `glm5` | `zai-org/GLM-5` | none |
| `gpt-2` | `openai-community/gpt2` | none |
| `phi-2` | `microsoft/phi-2` | none |
| `dbrx` | `allenai/OLMo-2-0425-1B` | none |
| `smaug-bpe` | `HuggingFaceTB/SmolLM3-3B` | none |
| `starcoder` | `bigcode/starcoder2-3b` | none |
| `tekken` | `mistralai/Mistral-Nemo-Base-2407` | none |
| `tekken-num3` | `mistralai/Mistral-Large-4-1T-A52B` | none |

`flint` has a `tokenizer.json` with no normalizer. `kimi-k2` uses a
tiktoken vocab with no normalizer. The other entries take the default.

## The snapshot test

`tests/load/data/llama_cpp_pre.json` is a snapshot of llama.cpp's table: the
pre type and flags of each name, the regex list of each pre type, and the
llama.cpp commit. `scripts/llama_pre_snapshot.py` writes it from a llama.cpp
checkout:

```sh
python scripts/llama_pre_snapshot.py ~/src/llama.cpp
```

`tests/load/test_tokenizer_pre_table.py` checks that every name in the
snapshot is in the table or in the test's `UNLISTED_PRES`, with a reason.
It checks that each entry has the snapshot's flags, and the snapshot's
regexes once both spellings are put in one form. When llama.cpp adds a
name, refresh the snapshot and the test shows the new name.

Parity is measured against the snapshot's llama.cpp commit, `e8df20d23`.
Older llama.cpp builds split Indic, Thai and Arabic text differently.

## Raw prompts and BOS

`/v1/completions` sends a raw prompt, with no chat template to write a BOS.
The route encodes it inside `raw_prompt_special_tokens` from
`gmlx/upstream/single_bos_fix.py`. Inside that block, every `prepare_inputs`
binding encodes with the tokenizer's special tokens and drops nothing. A
Gemma prompt then gets its BOS, as llama-server and Hugging Face give it.
When the text itself starts with BOS, the prompt holds two, as it does in
llama-server.

The chat routes do not use the block. Their templates write BOS, and the
single BOS rule in the same module keeps one.
