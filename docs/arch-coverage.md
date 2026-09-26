# Supported architectures

gmlx maps 41 GGUF architectures to model types, and 40 of them load end-to-end
with no `hf_source`. `scripts/check-coverage.py` generates this page from the
architecture table and the installed mlx-lm, so do not edit it by hand.

A GGUF loads when gmlx maps its `general.architecture` to a model type that has
a model class in an installed backend package and a config synthesizer exists
for the architecture. The backend is mlx-lm for most architectures and mlx-vlm
or mlx-embeddings for a few. Without a synthesizer, pass `hf_source` to supply
a config.json. The status column says which case applies:

| Status | Meaning |
|--------|---------|
| loadable | The model class and the synthesizer are both present. |
| needs synth (hf_source) | The model class is present and no synthesizer exists, so pass `hf_source`. |
| needs mlx-lm (or another backend) | The installed backend package has no model class for the architecture. |
| disabled (no known-good GGUF) | The code is complete, but each available GGUF is broken at conversion, so the loader refuses the architecture. |

The caveat column names the limit a user is most likely to hit, and an empty
cell means none is known. Each mapped GGUF architecture has one row, with its
family, backend, status and caveat:

| GGUF arch | Family | Backend | Status | Caveat |
|-----------|--------|---------|--------|--------|
| `deepseek2` | deepseek | mlx-lm | loadable | DeepSeek-V2, with softmax gating, is not supported. |
| `deepseek4` | deepseek | mlx-lm | loadable | The GGUF is not a llama.cpp conversion. The parity reference is the ds4 engine. |
| `deepseek41` | deepseek | mlx-lm | loadable | MTP is not yet implemented. The ds4 conversion's engram tables are read from the GGUF row by row. |
| `diffusion-gemma` | gemma | mlx-vlm | loadable |  |
| `ernie4_5-moe` | ernie | mlx-lm | loadable |  |
| `falcon-h1` | falcon | mlx-lm | loadable |  |
| `gemma` | gemma | mlx-lm | loadable |  |
| `gemma-embedding` | gemma | mlx-embeddings | loadable |  |
| `gemma2` | gemma | mlx-lm | loadable | Pass hf_source for 27B. Output matches the reference only within the 4096 window. |
| `gemma3` | gemma | mlx-lm | loadable | Pass hf_source for 27B. |
| `gemma3n` | gemma | mlx-lm | disabled (no known-good GGUF) |  |
| `gemma4` | gemma | mlx-lm | loadable |  |
| `glm-dsa` | deepseek | mlx-lm | loadable |  |
| `glm4` | glm | mlx-lm | loadable |  |
| `glm4moe` | glm | mlx-lm | loadable |  |
| `glm5next` | glm | mlx-lm | loadable | The hybrid cache cannot trim, so chat re-prefills on trim. |
| `gpt-oss` | gpt-oss | mlx-lm | loadable |  |
| `granite` | granite | mlx-lm | loadable |  |
| `granitehybrid` | granite | mlx-lm | loadable |  |
| `hunyuan-moe` | hunyuan | mlx-lm | loadable |  |
| `hy_v3` | hunyuan | mlx-lm | loadable | Early GGUFs whose arch is hy-v3 with a dash are not mapped. Reconvert them. |
| `hyv4` | hunyuan | mlx-lm | loadable | The GGUF has no MTP head, so speculative decoding is not available. `--kv-bits` is refused. |
| `kimi-k3` | kimi | mlx-lm | loadable | The hybrid cache cannot trim, so chat re-prefills on trim. |
| `llama` | llama | mlx-lm | loadable |  |
| `minimax-m2` | minimax | mlx-lm | loadable |  |
| `minimax-m3` | minimax_m3 | mlx-lm | loadable | Without indexer tensors or an indexer sidecar file, a GGUF runs dense. Its output is exact to 2048 tokens and degrades beyond that. |
| `mistral3` | llama | mlx-lm | loadable |  |
| `muse-glimmer` | muse | mlx-lm | loadable |  |
| `nemotron_h_moe` | nemotron | mlx-lm | loadable |  |
| `phi3` | phi3 | mlx-lm | loadable | Pass hf_source for the 128K long-context variants. |
| `qwen2` | qwen2 | mlx-lm | loadable |  |
| `qwen2moe` | qwen2 | mlx-lm | loadable |  |
| `qwen3` | qwen3 | mlx-lm | loadable |  |
| `qwen35` | qwen3 | mlx-lm | loadable |  |
| `qwen35moe` | qwen3 | mlx-lm | loadable |  |
| `qwen3moe` | qwen3 | mlx-lm | loadable |  |
| `qwen3next` | qwen | mlx-lm | loadable |  |
| `qwen3vlmoe` | qwen3 | mlx-lm | loadable |  |
| `qwen4exp` | qwen3 | mlx-lm | loadable | The MTP head needs the companion GGUF next to the target or --draft-gguf. |
| `seed_oss` | seed_oss | mlx-lm | loadable |  |
| `smollm3` | llama | mlx-lm | loadable |  |
