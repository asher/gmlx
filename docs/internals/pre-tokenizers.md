# Byte-level pre-tokenizers

gmlx builds a GGUF's tokenizer from the GGUF header. For a byte-level BPE
vocab, the header key `tokenizer.ggml.pre` names how text splits into words
before the merges run. gmlx follows llama.cpp for each name, so a prompt
gets the same token ids in both.

## The table

`PRE_TOKENIZERS` in `gmlx/load/tokenizer.py` maps each name to an entry with
three columns:

| Column | Meaning |
|--------|---------|
| `patterns` | The split regexes, applied in sequence. Each regex splits every piece that the one before it made. |
| `ignore_merges` | A word that is a whole vocab entry becomes one id, and the merges do not run on it. |
| `add_bos` | The BOS default for a GGUF without `tokenizer.ggml.add_bos_token`. |

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

## Unknown names

A name that the table does not list falls back to the Qwen 2 split. The
load prints a warning that names the pre-tokenizer, because token ids can
then differ from llama.cpp. An SPM vocab, such as Gemma's, has no word split
and gets no warning.

## Unicode normalization

gmlx applies NFC to every byte-level vocab except `kimi-k2` and `flint`.
llama.cpp applies no normalization to a BPE vocab. The ids therefore differ
for text that is not already in NFC, such as an `e` followed by a combining
accent.

This is a known defect. The reference for each family is its
`tokenizer.json`. Qwen declares an NFC normalizer, so gmlx is right there.
Llama 3, Tekken, GPT-2 and SmolLM declare none, so gmlx is wrong there on
decomposed text.

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
