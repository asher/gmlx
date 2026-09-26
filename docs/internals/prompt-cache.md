# Prompt cache internals

The prompt cache chooses its tiers for each architecture, exposes counters
that show whether reuse works, and has environment switches that tune or
disable each layer. Operators read [Prompt cache](../prompt-cache.md).

## Which tier serves which architecture

The cache routes each model by its cache shape once, at load, and logs the
routing as `APC tier:` so that a silent mis-route is visible. The tiers
differ in storage layout, not in whether hits happen, so every shape in the
table gets reuse except one. A MiniMax-M3 file with its sparse-attention
indexer armed, the MSA row, gets none. In the table, [GDN](../glossary.md#gdn)
is the gated delta network recurrence of the Qwen3.5 and 3.6 hybrids, and
a CacheList model is one whose layers each hold several caches of different
kinds, the mlx-lm class of that name.

| Cache shape | Example archs | Tier |
|-------------|---------------|------|
| Plain KV, dense or MoE | llama, qwen2, qwen3, qwen3moe, glm4, glm4moe, deepseek2, phi3, granite, hunyuan-moe, minimax-m2, gemma2 | `block` |
| Hybrid GDN, recurrent plus KV layers | qwen35, qwen35moe, qwen3next, kimi-k3, nemotron_h_moe, granitehybrid | `ckpt` |
| Sliding-window attention | gemma3, gemma4, gpt-oss | `ckpt` |
| CacheList or pure recurrent | falcon-h1, deepseek4 | `exact` |
| MSA indexer armed | minimax-m3 indexer GGUFs | None, with a logged warning. The indexless GGUF serves through the block tier. |

## The cache layers

A request passes through five layers in order, and all of them are on by
default. The prefix layer and the drafter sidecar exist only for
speculative models, and checkpoints only for checkpoint-tier models.

- Prefix layer: An in-memory LRU holds post-prefill KV and hidden state. A
  request sharing a prefix with an earlier one skips that prefill even with
  `cache:` off.
- Shared pools: With `cache.enabled`, the lookup order of exact, block and
  disk fills the prompt cache before prefill, including warm restarts from the
  SSD tier.
- Retirement: At request finish the whole sequence, prompt plus reply, is
  stored back, so the next turn of a conversation warm-starts past the whole
  of this one.
- Drafter sidecar: A native MTP head keeps a separate KV, and a warm target
  paired with an empty drafter KV decodes at degraded acceptance until that
  KV is rebuilt. A small sidecar entry therefore saves the drafter's KV
  beside the target's, so a warm hit restores both.
- Checkpoints: Hybrid models save restore points piecewise along a prefill
  and while generating, plus targeted ones at the end of the system prompt,
  one token before the prompt end and at the predicted next-turn boundary.
  The system-prompt one is what lets parallel agents sharing a prompt
  restore from it, and a checkpoint's copy on the disk tier is called its
  skeleton. Because a batched prefill takes no checkpoints, prompt prefill
  on these models runs one request at a time.

## Checkpoint-tier counters

`GET /v1/cache/stats` carries the `ckpt_*` fields for each model. They
change only on checkpoint-tier architectures, so all-zero on a block-tier
or exact-tier model is normal. On a checkpoint-tier model they are the
fields to judge prefix reuse from. Ratios built on the stock counters
mislead there, because a checkpoint lookup increments the shared hit
counters on success but records nothing on a miss, and the token totals
include window snapshots that can never be shared.

`disk_writes` counts write operations. Each exact-format entry counts one,
checkpoint skeletons and drafter sidecars included, and a block shard counts
one for each of its blocks.

| Field | Meaning |
|-------|---------|
| `ckpt_stores` | Each prefix saved for reuse adds one. Zero after a few requests on a checkpoint-tier model means nothing is being cached. |
| `ckpt_hits`, `ckpt_matched_tokens` | These count warm starts from a saved prefix and the prompt tokens they skipped. Matched tokens approach total prompt tokens on repeat-heavy traffic. |
| `ckpt_declines` | Each save the server skips counts under its reason. Occasional entries are normal. All requests under one reason means reuse is not working for that traffic. |
| `ckpt_missed_adoptions` | Each request that matched a saved prefix but could not use it adds one. The count stays 0 in healthy operation, and growth indicates a bug. |
| `ckpt_pool_evictions` | Each prefix discarded to make room adds one. Evictions are normal on long sessions, most of all on sliding-window models. Raise `num_blocks` to keep more. |
| `ckpt_skeleton_writes` | Each save mirrored to the disk tier for warm restarts adds one. Zero with `disk` on means a restart will start cold. |
| `sidecar_writes` | Each draft-model cache entry saved next to its target entry adds one. Only speculative decoding writes them. |
| `retire_fallback_suppressed` | A retirement store is skipped and counted here when the predicted next-turn render has diverged, so the entry could never match. The turn checkpoint covers it. |

The server watches for two failures and warns once per model about each. One
warning fires when more than `GMLX_APC_CKPT_TRIPWIRE` requests have
completed with zero stores. The other fires when that many lookups have
matched a saved prefix but adopted nothing, with zero hits. Either warning
means prefix reuse is not working for that model, so file an issue with the
`/v1/cache/stats` snapshot.

## Under kvarn KV

Tier routing matches fp16 KV with one change. Dense models use the exact
tier, since the 16-token block tier cannot split kvarn's 128-token records.
Checkpoint-shaped stacks, which are the hybrid-GDN and sliding-window
families in the reuse table whose attention head_dim is 128, 256 or 512,
keep full checkpoint-tier reuse and store kvarn records. The attention
payload lives inline in the record and not in pool blocks, so
`GMLX_APC_CKPT_BUDGET_MB` bounds the tier's memory and `APC_NUM_BLOCKS`
matters little. Entries and disk skeletons are keyed to the kvarn width and
tail, so a config change or a restart that switches between stock and kvarn
misses instead of reading a stale format. Cascade shared-prefix decode is
off under kvarn and logs that once. Speculative rollback into a sealed
record reopens it from its codes, one lossy round trip, while rows still in
the fp16 tail roll back exactly. A batch of kvarn rows snapshots each row's
own start, end and tail window with its buffers, and a restore from an
older batch layout refuses instead of misreading it.

## Environment switches

| Variable | Meaning |
|----------|---------|
| `GMLX_SPEC_APC` | This is the master switch. `0` turns all speculative cache layers off at once, including lookups, stores, the sidecar and the checkpoint tier. |
| `GMLX_SPEC_APC_RETIRE` | `0` turns off only the retirement store. |
| `GMLX_SPEC_APC_SIDECAR` | `0` turns off only the drafter-KV sidecar. |
| `GMLX_SPEC_APC_CKPT` | `0` turns off only the hybrid checkpoint tier. The exact full-clone path is used instead. |
| `GMLX_SPEC_APC_ENTRIES` | The prefix layer keeps this many LRU entries. The default is `4`. |
| `GMLX_SPEC_APC_SIDECAR_ENTRIES` | The drafter sidecar keeps this many LRU entries. The default is `12`. |
| `GMLX_SPEC_APC_BUDGET_MB` | The in-memory prefix layer stays within this many MB. The default is `8192`. |
| `GMLX_SPEC_APC_SIDECAR_BUDGET_MB` | The drafter-sidecar LRU stays within this many MB. The default is `512`. |
| `GMLX_APC_STORE_EVAL_CHUNK` | The post-prefill store evaluates this many blocks per step, which bounds the prefill-thread stall on long prompts. The default is `32`. |
| `GMLX_APC_CKPT_INTERVAL` | Prefill checkpoints fall this many tokens apart, rounded to the chunk grid. The default is `4096`, and `0` saves only the final checkpoint. |
| `GMLX_APC_CKPT_REPLAY` | `0` disables the replay checkpoint, so identical resends prefill cold again. |
| `GMLX_APC_CKPT_REPLAY_MIN` | A replay checkpoint on a recurrent model needs a prompt of at least this many tokens. The default is `1024`, since shorter prompts re-prefill quickly. |
| `GMLX_APC_CKPT_TURN` | `0` disables the turn checkpoint, so next-turn reuse falls back to the interval grid. |
| `GMLX_APC_CKPT_SYS` | `0` disables the system-prompt anchor on both tiers. Sibling requests sharing a system prompt then prefill the shared prefix without cache reuse. |
| `GMLX_APC_CKPT_SYS_MIN` | An anchor needs at least this many tokens of shared system prefix. The default is `256`, raised to the replay minimum on recurrent models. |
| `GMLX_APC_ANCHOR_ENTRIES` | Exact-mode models keep system-prompt anchors as whole-prefix clones in an LRU of this many entries. The default is `4`. |
| `GMLX_APC_ANCHOR_BUDGET_MB` | The exact-tier anchor LRU stays within this many MB, default `4096`, and never evicts its newest entry. A long shared prefix on a pooling stack clones to GBs. |
| `GMLX_APC_CKPT_TRIPWIRE` | The not-storing check warns after this many requests, and the not-hitting check after this many unusable matches. The default is `5`, and `0` silences both. |
| `GMLX_APC_CKPT_RECORDS` | The checkpoint-record LRU keeps this many entries. The default is `32`. |
| `GMLX_APC_CKPT_BUDGET_MB` | Checkpoint payload stays within this many MB. The default is `4096`, and resident memory grows toward it on hybrid models under multi-turn traffic. |
| `GMLX_APC_DECODE_CKPT` | Hybrid models snapshot at this interval in generated tokens, anchored to the prompt end. The default is `512`, widening with context. `0` turns it off. |
| `GMLX_APC_RETIRE_LCP` | `0` keys retirement on the forwarded ids instead of the predicted next-turn render. This also disables decode-time snapshots, which key on the prediction. |
| `GMLX_APC_FRESH_WAIT_MS` | The freshness admission gate admits siblings arriving together one at a time, holding each at most this many ms. The default is `500`, and `0` disables it. |
| `GMLX_APC_FRESH_MIN` | The gate holds a sibling only past this many uncovered shared-prefix tokens. The default is `256`. Below it, a duplicate prefill beats the wait. |

`GMLX_FAITHFUL_HISTORY`, which restores mlx-vlm's stock chat-history
rebuild, is a user-facing switch and is documented in
[Environment variables](../env-vars.md).
