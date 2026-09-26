# Family defaults

gmlx reads each model's family from its GGUF header and starts every
request from the sampling values that the family's publisher recommends.
Some families also publish other operating points, which gmlx offers as
intents such as `@coding` and `@reasoning-high`. `gmlx profiles` prints
both for every family, and with a model id it prints that model's
resolved values.

```sh
gmlx profiles                     # every family and its intents
gmlx profiles qwen3.8-27b-ud-q6   # the resolved values of one configured model
```

Each family covers the architectures in its row's GGUF architectures column. An intent
is shown with the family values it keeps, and an intent that a family does
not define gives the family defaults. The reasoning level goes by three
names. It is `reasoning_effort` for gpt-oss, DeepSeek-V4.1, Hy3 and HY4,
`thinking_effort` for Kimi K3, and `reasoning_strength` for Muse. Each value
comes from the model card or generation config cited in
`gmlx/gen/profiles.py`, and HY4's values come from its GGUF. The
`default` row holds generic values that no model card backs.

| Family | GGUF architectures | Base values | Intents |
|--------|-------------|--------------------|----------------|
| `qwen3.6` | `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp` | temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 | `@coding`: temperature=0.6 top_p=0.95 top_k=20 min_p=0.0; `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 presence_penalty=1.5 enable_thinking=False |
| `qwen3` | `qwen3`, `qwen3moe`, `qwen3vlmoe` | temperature=0.6 top_p=0.95 top_k=20 min_p=0.0 | `@instruct`: temperature=0.7 top_p=0.8 top_k=20 min_p=0.0 enable_thinking=False |
| `qwen2.5` | `qwen2`, `qwen2moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `gemma` | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `diffusion-gemma` | temperature=1.0 top_p=0.95 top_k=64 | - |
| `gpt-oss` | `gpt-oss` | temperature=1.0 top_p=1.0 | `@reasoning-high`: temperature=1.0 top_p=1.0 reasoning_effort=high; `@reasoning-low`: temperature=1.0 top_p=1.0 reasoning_effort=low; `@reasoning-medium`: temperature=1.0 top_p=1.0 reasoning_effort=medium |
| `glm` | `glm4`, `glm4moe`, `glm-dsa`, `glm5next` | temperature=1.0 top_p=0.95 | - |
| `deepseek` | `deepseek2`, `deepseek4` | temperature=0.6 top_p=0.95 | - |
| `deepseek41` | `deepseek41` | temperature=1.0 top_p=0.95 | `@reasoning-high`: temperature=1.0 top_p=0.95 reasoning_effort=high; `@reasoning-low`: temperature=1.0 top_p=0.95 reasoning_effort=low; `@reasoning-max`: temperature=1.0 top_p=0.95 reasoning_effort=max |
| `minimax` | `minimax-m2`, `minimax-m3` | temperature=1.0 top_p=0.95 top_k=40 | - |
| `nemotron` | `nemotron_h_moe` | temperature=1.0 top_p=0.95 | - |
| `hunyuan` | `hunyuan-moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `hy3` | `hy_v3` | temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> | `@reasoning-high`: temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> reasoning_effort=high; `@reasoning-low`: temperature=0.9 thinking_start_token=<think:opensource> thinking_end_token=</think:opensource> reasoning_effort=low |
| `hy4` | `hyv4` | temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> | `@reasoning-high`: temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> reasoning_effort=high; `@reasoning-low`: temperature=0.9 top_p=1.0 thinking_start_token=<think:6124c78e> thinking_end_token=</think:6124c78e> reasoning_effort=low |
| `kimi` | `kimi-k3` | temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> | `@reasoning-high`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=high; `@reasoning-low`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=low; `@reasoning-max`: temperature=1.0 top_p=0.95 thinking_start_token=<|open|>think<|sep|> thinking_end_token=<|close|>think<|sep|> thinking_effort=max |
| `kimi-k2` | `deepseek2 named (?i)\bkimi` | temperature=1.0 top_p=0.95 | - |
| `muse` | `muse-glimmer` | temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> | `@reasoning-high`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=high; `@reasoning-low`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=low; `@reasoning-medium`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=medium; `@reasoning-xhigh`: temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=<|start|>assistant to=self<|message|> thinking_end_token=<|eom|> reasoning_strength=xhigh |
| `llama` | `llama`, `smollm3` | temperature=0.6 top_p=0.9 | - |
| `mistral` | `mistral3` | temperature=0.15 | - |
| `default` | (anything else) | temperature=0.7 top_p=0.95 | `@coding`: temperature=0.3 top_p=0.95; `@creative`: temperature=1.0 top_p=0.95 min_p=0.05; `@instruct`: temperature=0.7 top_p=0.95 |

[How a request gets its settings](config.md#how-a-request-gets-its-settings)
explains how the defaults combine with your own profiles, and
[`models.*.family`](config.md#modelsfamily) replaces one model's detected
family.
