# Prompt cache internals

The prompt cache chooses its tier for each architecture and exposes
counters that show whether reuse works. The user guide is
[Prompt cache](../prompt-cache.md).

## Which tier serves which architecture

The cache routes each model by its cache stack's shape once, at the first
probe, and logs the result as `APC tier:`, or as
`APC OFF for this model` when no tier serves it. That log line is the
source of truth for a model. `model_apc_mode` in `mlx_vlm.apc` picks the
block or exact tier, and `_ckpt_active` in `gmlx/spec/engine.py` moves an
exact-mode stack with gated-delta or sliding-window layers to the
checkpoint tier.

| Cache shape | Example | Tier |
|-------------|---------|------|
| Plain KV, dense or MoE | llama | `block` |
| Recurrent layers with KV layers | qwen35 | `ckpt` |
| Sliding-window attention | gemma4 | `ckpt` |
| CacheList or pure recurrent | falcon-h1 | `exact` |
| A stack that no tier can split | minimax-m3 with its indexer | None, with a logged warning. |

Hits happen on every tier, and the tiers differ in storage layout. Under
kvarn KV, a dense model uses the exact tier, because the block tier cannot
split kvarn's 128-token records. A checkpoint-tier stack keeps its
tier and stores kvarn records. Entries and disk skeletons are keyed to the
kvarn width and tail, so a restart that switches between stock and kvarn
misses instead of reading a stale format.

## The cache layers

A request passes through five layers in order. Their switches are all on
by default, but only the prefix layer works with `cache:` off, and the
other four need `cache.enabled`. The prefix layer and the drafter sidecar
exist only for speculative models, and checkpoints only for
checkpoint-tier models.

- Prefix layer: An in-memory LRU holds post-prefill KV and hidden state. A
  request that shares a prefix with an earlier one skips that prefill.
- Shared pools: The block, exact and disk lookups fill the prompt cache
  before prefill, so a request can warm-start from memory or from the SSD
  tier. The longest match wins.
- Retirement: At request finish the whole sequence, prompt plus reply, is
  stored back, so a conversation's next turn warm-starts past the whole
  finished turn.
- Drafter sidecar: A native MTP head keeps a separate KV. A warm target
  with an empty drafter KV decodes at degraded acceptance, so a small
  sidecar entry saves the drafter KV beside the target entry, and a warm
  hit restores both.
- Checkpoints: Checkpoint-tier models save checkpoints along a prefill and
  while generating, plus targeted ones at the end of the system prompt,
  one token before the prompt end and at the predicted next-turn boundary.
  The system-prompt checkpoint lets parallel agents that share a prompt
  restore from it. A batched prefill takes no checkpoints, so prompt
  prefill on these models runs one request at a time.

Each layer's switches are listed in the prompt cache section of
[Debug switches](debug-switches.md#prompt-cache).

## Checkpoint-tier counters

`GET /v1/cache/stats` carries the `ckpt_*` fields for each model. They
change only on checkpoint-tier architectures, so all zeros on a block-tier
or exact-tier model is normal. On a checkpoint-tier model they are the
fields to judge prefix reuse from. Ratios built on the stock counters
mislead there, because a checkpoint lookup counts a hit on success but
records nothing on a miss.

This table gives each counter and what a change in it means.

| Field | Meaning |
|-------|---------|
| `ckpt_stores` | Each prefix saved for reuse adds one. Zero after a few requests on a checkpoint-tier model means nothing is being cached. |
| `ckpt_hits`, `ckpt_matched_tokens` | These count warm starts from a saved prefix and the prompt tokens they skipped. |
| `ckpt_declines` | Each save the server skips counts under its reason. When every request lands under one reason, reuse does not work for that traffic. |
| `ckpt_missed_adoptions` | Each request that matched a saved prefix but could not use it adds one. Any growth indicates a bug. |
| `ckpt_pool_evictions` | Each prefix discarded to make room adds one. Evictions are normal on long sessions. |
| `ckpt_skeleton_writes` | Each save mirrored to the disk tier adds one. Zero with `disk` on means a restart starts cold. |
| `sidecar_writes` | Each drafter cache entry saved beside its target entry adds one. |
| `retire_fallback_suppressed` | Each retirement store skipped because the predicted next-turn render diverged adds one. The turn checkpoint covers it. |

The server warns once per model when requests armed for checkpoints store
nothing, and once when lookups match a saved prefix but adopt nothing.
Either warning means prefix reuse does not work for that model, so file
an issue with the `/v1/cache/stats` snapshot.
