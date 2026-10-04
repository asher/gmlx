# Family defaults

gmlx reads each model's family from its GGUF header and starts every
request from the sampling values that the family's publisher recommends.
Some families also publish values for a task, which gmlx offers as intents
such as `@coding` and `@reasoning-high`.

```sh
gmlx profiles                     # every family and its intents
gmlx profiles qwen3.8-27b-ud-q6   # the resolved values of one configured model
```

The table lists the base values of each family and what each of its intents
changes. An intent that a family does not list runs with the base values.
The values come from the publishers' model cards and generation configs.
The `default` row holds generic values for any other architecture.

Some GGUFs carry their publisher's sampling values in the header, as
`general.sampling.*` keys. Each of these replaces the family's value, and
your own profiles still win over them. `gmlx profiles <id>` shows the values
a model ends up with.

| Family | GGUF architectures | Base values | Intents |
|--------|-------------|--------------------|----------------|
| `qwen3.6` | `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp` | temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 | `@coding`: temperature=0.6; `@instruct`: temperature=0.7 top_p=0.8 presence_penalty=1.5 enable_thinking=False |
| `qwen3` | `qwen3`, `qwen3moe`, `qwen3vlmoe` | temperature=0.6 top_p=0.95 top_k=20 min_p=0.0 | `@instruct`: temperature=0.7 top_p=0.8 enable_thinking=False |
| `qwen2.5` | `qwen2`, `qwen2moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `gemma` | `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `diffusion-gemma` | temperature=1.0 top_p=0.95 top_k=64 | - |
| `gpt-oss` | `gpt-oss` | temperature=1.0 top_p=1.0 | `@reasoning-high`: reasoning_effort=high; `@reasoning-low`: reasoning_effort=low; `@reasoning-medium`: reasoning_effort=medium |
| `glm` | `glm4`, `glm4moe`, `glm-dsa`, `glm5next` | temperature=1.0 top_p=0.95 | - |
| `deepseek` | `deepseek2`, `deepseek4` | temperature=0.6 top_p=0.95 | - |
| `deepseek41` | `deepseek41` | temperature=1.0 top_p=0.95 | `@reasoning-high`: reasoning_effort=high; `@reasoning-low`: reasoning_effort=low; `@reasoning-max`: reasoning_effort=max |
| `minimax` | `minimax-m2`, `minimax-m3` | temperature=1.0 top_p=0.95 top_k=40 | - |
| `nemotron` | `nemotron_h_moe` | temperature=1.0 top_p=0.95 | - |
| `hunyuan` | `hunyuan-moe` | temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 | - |
| `hy3` | `hy_v3` | temperature=0.9 thinking_start_token=&lt;think:opensource&gt; thinking_end_token=&lt;/think:opensource&gt; | `@reasoning-high`: reasoning_effort=high; `@reasoning-low`: reasoning_effort=low |
| `hy4` | `hyv4` | temperature=0.9 top_p=1.0 thinking_start_token=&lt;think:6124c78e&gt; thinking_end_token=&lt;/think:6124c78e&gt; | `@reasoning-high`: reasoning_effort=high; `@reasoning-low`: reasoning_effort=low |
| `kimi` | `kimi-k3` | temperature=1.0 top_p=0.95 thinking_start_token=&lt;\|open\|&gt;think&lt;\|sep\|&gt; thinking_end_token=&lt;\|close\|&gt;think&lt;\|sep\|&gt; | `@reasoning-high`: thinking_effort=high; `@reasoning-low`: thinking_effort=low; `@reasoning-max`: thinking_effort=max |
| `kimi-k2` | `deepseek2 named (?i)\bkimi` | temperature=1.0 top_p=0.95 | - |
| `muse` | `muse-glimmer` | temperature=1.0 top_p=0.95 top_k=64 thinking_start_token=&lt;\|start\|&gt;assistant to=self&lt;\|message\|&gt; thinking_end_token=&lt;\|eom\|&gt; | `@reasoning-high`: reasoning_strength=high; `@reasoning-low`: reasoning_strength=low; `@reasoning-medium`: reasoning_strength=medium; `@reasoning-xhigh`: reasoning_strength=xhigh |
| `llama` | `llama`, `smollm3` | temperature=0.6 top_p=0.9 | - |
| `mistral` | `mistral3` | temperature=0.15 | - |
| `default` | (anything else) | temperature=0.7 top_p=0.95 | `@coding`: temperature=0.3; `@creative`: temperature=1.0 min_p=0.05 |

[How a request gets its settings](config.md#how-a-request-gets-its-settings)
explains how the defaults combine with your own profiles, and
[`models.*.family`](config.md#modelsfamily) replaces one model's detected
family.
