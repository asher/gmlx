# Prompt cache

The prompt cache lets the server skip prefill for the start of a prompt
that it has seen before. It is off by default, reuses more or less of a
prompt depending on the model family, and can keep entries on the SSD.

Agents and long chats gain the most, because they send the same system
prompt and history again on every turn. With the cache, a 32K-token
history that the server has already read costs well under a second before
the first token, instead of tens of seconds of prefill.

- [Turning it on](#turning-it-on)
- [What each family reuses](#what-each-family-reuses)
- [The SSD tier](#the-ssd-tier)
- [What a hit restores](#what-a-hit-restores)

## Turning it on

The cache is off unless the configuration file turns it on, and
`gmlx init` writes a file with it on:

```yaml
server:
  cache:
    enabled: true
```

[`server.cache`](config.md#prompt-cache) holds the settings, and a `cache`
block in a profile or in a model's `overrides` changes them for some
models. The server chooses the block size and the size of the pool for
each model. `GET /v1/cache/stats` reports the hits, the stores and the
other counters, as [Endpoints](api.md#endpoints) lists.

## What each family reuses

How much of a prompt the cache can reuse depends on the model family,
because the state of a recurrent layer cannot be rolled back to an
earlier token. [GDN](glossary.md#gdn), the gated delta network, is the
recurrent layer of the Qwen3.5, Qwen3.6 and Qwen3.8 hybrids.

| Family | Identical prompt | Next turn | Edited or regenerated turn |
|--------|------------------|-----------|----------------------------|
| Dense models and MoE models with plain attention | All of it. | All of the earlier conversation. | Up to the edit, in blocks of 16 to 256 tokens. |
| GDN hybrids, such as Qwen3.5 and Qwen3.6 | All but the last token. | Up to the turn boundary, rounded down to the nearest restore point. | Up to the nearest restore point before the edit. |
| Sliding-window models, such as gemma-4 and gpt-oss | As for GDN hybrids. | Up to a few tokens before the change, once the prompt is longer than the window. | As for GDN hybrids. |
| Pure recurrent and multi-cache models, such as Falcon-H1 and DeepSeek-V4 | All of it. | All of it, because each turn extends the stored conversation unchanged. | None. An edited conversation prefills from the start. |

Hybrid models save restore points along the prompt and at the end of each
turn. A prompt shorter than 1024 tokens saves no restore point for an
identical resend, because it prefills quickly anyway.

A sliding-window model under speculative decoding keeps no record of the
tokens it generated, so the next turn reuses only up to the end of the
previous prompt, and the reply is read again. Under a
[kvarn KV cache](kv-quantization.md), dense models reuse whole prompts
only, as the pure recurrent models do, because the cache blocks cannot
split kvarn records.

Some thinking models remove the reasoning of earlier turns when the chat
template renders the conversation again. The rendered text then changes
right after the start of the last reply, so the server stores the entry
for the reply in the form that the next turn will render. What comes after
the change is read again, on this server as on any other, because the
template itself changes the text.

## The SSD tier

The SSD tier keeps entries across model unloads and server restarts, and
holds more entries than memory would. `gmlx init --disk-cache` writes a
file with it on, or you can add it to the file:

```yaml
server:
  cache:
    enabled: true
    disk: {path: ~/.cache/gmlx/apc, max_gb: 100}
```

Its place and size are the [`cache.disk`](config.md#cachedisk) keys, and
the server removes entries to keep the tier within that size.

## What a hit restores

A hit restores more than the KV cache of the prompt. The server stores the
finished conversation again after the reply, so the next turn starts after
the reply. For a model with speculative decoding, it also stores the KV
cache of the drafter beside the model's. [Prompt cache
internals](internals/prompt-cache.md) describes the tiers, their counters
and the switches for finding problems.
