# Prompt cache internals

How the prompt cache tiers are chosen per architecture, the counters that
show whether reuse is working and the environment switches that tune or
disable each layer. This page is for contributors, while operators read
[performance.md](../performance.md#the-prompt-cache). Upstream calls the
cache APC, and the switches use that name.

## Which tier serves which architecture

The cache routes each model by its cache shape once, at load, and logs the
routing as `APC tier:` so that a silent mis-route is visible. The tiers
differ in storage layout, not in whether hits happen, so every shape in the
table gets reuse except the armed MSA indexer, which gets none.

| Cache shape | Example archs | Tier |
|-------------|---------------|------|
| plain KV, dense or MoE | llama, qwen2/3, qwen3moe, glm4(-moe), deepseek2/v3, phi3, granite, hunyuan, minimax-m2, gemma2 | block |
| hybrid GDN, recurrent plus KV layers | qwen3.5/3.6 dense and MoE, qwen3-next, kimi-k3, nemotron-h, granitemoehybrid | ckpt |
| sliding-window attention | gemma3, gemma-4 including E4B and 3n, gpt-oss, SWA llama | ckpt |
| CacheList or pure recurrent | falcon-h1, mamba2, rwkv7, plamo2, deepseek-v3.2, deepseek4 | exact |
| MSA indexer armed | minimax-m3 indexer GGUFs | none, with a logged warning. The indexless GGUF serves through the block tier. |

## The cache layers

A request passes through the layers below in order, and all of them are on
by default. The first three serve every model, and the last two exist for
speculative and checkpoint-tier models.

- Prefix layer. An in-memory LRU of post-prefill KV and hidden state. A
  request sharing a prefix with an earlier one skips that prefill even with
  `cache:` off.
- Shared pools. With `cache.enabled`, the lookup order of exact, block and
  disk fills the prompt cache before prefill, including warm restarts from the
  SSD tier.
- Retirement. At request finish the whole sequence, prompt plus reply, is
  stored back, so the next turn of a conversation warm-starts past the whole
  of this one.
- Drafter sidecar. A native MTP head keeps a separate KV, and a warm target
  paired with an empty drafter KV decodes at degraded acceptance until that
  KV is rebuilt. A small sidecar entry therefore saves the drafter's KV
  beside the target's, so a warm hit restores both.
- Checkpoints. Hybrid models save restore points piecewise along a prefill
  and while generating, plus targeted ones at the end of the system prompt,
  one token before the prompt end and at the predicted next-turn boundary.
  The system-prompt one is what lets parallel agents sharing a prompt
  restore from it. A checkpoint's disk copy is its skeleton. Prompt prefill
  on these models runs one request at a time.

## Checkpoint-tier counters

`GET /v1/cache/stats` carries the `ckpt_*` fields for each model, but they
only change on checkpoint-tier architectures, so all-zero on a block-tier
or exact-tier model is normal. On checkpoint-tier models they answer
whether prefix reuse is working, and reuse should be judged from these
fields rather than from ratios built on the stock ones, because checkpoint
lookups increment the shared hit counters on success but record nothing on
a miss, and the token totals include window snapshots that can never be
shared. `disk_writes` counts write operations, one for each exact-format
entry, checkpoint skeletons and drafter sidecars included, plus one for
each block of a block shard.

| Field | What it tells you |
|-------|-------------------|
| `ckpt_stores` | Prefixes saved for reuse. Still zero after a few requests on a checkpoint-tier model means nothing is being cached. The server logs a one-time warning. |
| `ckpt_hits`, `ckpt_matched_tokens` | Requests that warm-started from a saved prefix, with the prompt tokens they skipped. On repeat-heavy traffic matched tokens approach total prompt tokens. |
| `ckpt_declines` | Saves the server skipped, by reason. Occasional entries are normal. All requests counted under one reason means reuse is not working for that traffic. |
| `ckpt_missed_adoptions` | Requests that matched a saved prefix but could not use it. Stays 0 in healthy operation. Growth indicates a bug and triggers a one-time warning. |
| `ckpt_pool_evictions` | Saved prefixes discarded to make room. Normal on long sessions, fastest on sliding-window models. Raise `num_blocks` to keep more history cached. |
| `ckpt_skeleton_writes` | Saves mirrored to the disk tier for warm restarts. Zero with `disk` on means a restart will start cold. |
| `sidecar_writes` | Draft-model cache entries saved next to their target entries. Speculative decoding only. |
| `retire_fallback_suppressed` | Whole-sequence retirement stores skipped because the predicted next-turn render had diverged, so the entry could never match. The turn checkpoint covers them. |

The server also checks for a tier that is not storing or hitting and warns
once per model, after `GMLX_APC_CKPT_TRIPWIRE` completed requests with zero
stores, or that many unusable matches with zero hits, with a default of 5.
Either warning means prefix reuse is not working for that model, so file an
issue with the `/v1/cache/stats` snapshot.

## Under kvarn KV

Tier routing matches fp16 KV with one change: dense models use the exact
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
the fp16 tail roll back exactly.

## Environment switches

| Variable | Meaning |
|----------|---------|
| `GMLX_SPEC_APC` | The master switch. `0` turns all speculative cache layers off at once, meaning lookups, stores, the sidecar and the checkpoint tier. |
| `GMLX_SPEC_APC_RETIRE` | `0` turns off just the retirement store. |
| `GMLX_SPEC_APC_SIDECAR` | `0` turns off just the drafter-KV sidecar. |
| `GMLX_SPEC_APC_CKPT` | `0` turns off just the hybrid checkpoint tier. The exact full-clone path is used instead. |
| `GMLX_SPEC_APC_ENTRIES` | Prefix-layer LRU entries. Default `4`. |
| `GMLX_SPEC_APC_SIDECAR_ENTRIES` | Drafter-sidecar LRU entries. Default `12`. |
| `GMLX_SPEC_APC_BUDGET_MB` | Byte budget for the in-memory prefix layer, in MB. Default `8192`. |
| `GMLX_SPEC_APC_SIDECAR_BUDGET_MB` | Byte budget for the drafter-sidecar LRU, in MB. Default `512`. |
| `GMLX_APC_STORE_EVAL_CHUNK` | Blocks evaluated in each step of the post-prefill store. Default `32`. Bounds the prefill-thread stall on long prompts. |
| `GMLX_APC_CKPT_INTERVAL` | Prefill checkpoint interval in tokens, rounded to the chunk grid. Default `4096`. `0` saves only the final checkpoint. |
| `GMLX_APC_CKPT_REPLAY` | `0` disables the replay checkpoint, so identical resends prefill cold again. |
| `GMLX_APC_CKPT_REPLAY_MIN` | Minimum prompt tokens before a replay checkpoint is saved on recurrent models. Default `1024`. Shorter prompts re-prefill quickly. |
| `GMLX_APC_CKPT_TURN` | `0` disables the turn checkpoint, so next-turn reuse falls back to the interval grid. |
| `GMLX_APC_CKPT_SYS` | `0` disables the system-prompt anchor on both tiers. Sibling requests sharing a system prompt then prefill the shared prefix without cache reuse. |
| `GMLX_APC_CKPT_SYS_MIN` | Minimum tokens of shared system prefix before an anchor is saved. Default `256`, raised to the replay minimum on recurrent models. |
| `GMLX_APC_ANCHOR_ENTRIES` | Exact-tier anchor LRU entries, where exact-mode models keep system-prompt anchors as whole-prefix clones. Default `4`. |
| `GMLX_APC_ANCHOR_BUDGET_MB` | Byte budget for the exact-tier anchor LRU, in MB. Default `4096`. A long shared prefix on a pooling stack clones to GBs. The newest entry is never evicted. |
| `GMLX_APC_CKPT_TRIPWIRE` | Completed requests before the not-storing and not-hitting checks warn. Default `5`. `0` silences both. |
| `GMLX_APC_CKPT_RECORDS` | Checkpoint-record LRU entries. Default `32`. |
| `GMLX_APC_CKPT_BUDGET_MB` | Byte budget for checkpoint payload, in MB. Default `4096`. Resident memory grows toward it on hybrid models under multi-turn traffic. |
| `GMLX_APC_DECODE_CKPT` | Decode-time snapshot interval in generated tokens on hybrid models, anchored to the prompt end. Default `512`. `0` turns it off. Widens with context. |
| `GMLX_APC_RETIRE_LCP` | `0` keys retirement on the forwarded ids instead of the predicted next-turn render. This also disables decode-time snapshots, which key on the prediction. |
| `GMLX_APC_FRESH_WAIT_MS` | Maximum hold for the freshness admission gate, in ms. Default `500`. `0` disables it. Siblings arriving together admit one at a time. |
| `GMLX_APC_FRESH_MIN` | Minimum uncovered shared-prefix tokens before the gate holds a sibling. Default `256`. Below it the duplicate prefill takes less time than the wait. |
| `GMLX_FAITHFUL_HISTORY` | `0` restores mlx-vlm's stock chat-history rebuild, which drops `reasoning_content` from non-tool assistant messages before the template sees it. |
