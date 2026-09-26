# Debug switches

Debug switches are environment variables that change how gmlx builds or
routes a model, or make it report what it did, so that an A/B run can
isolate a defect. The user-facing variables are in
[Environment variables](../env-vars.md).

The disabling switches slow the model or turn a fix off. Most logging
switches cost nothing but output, but the layer profile evaluates after
each layer part and slows the run it measures. A switch that takes a
number moves a threshold or a width, so that a run can test a component
at more than one setting. The module that reads a switch holds its
default.

## Contents

- [Speculative decoding](#speculative-decoding)
- [Prompt cache](#prompt-cache)
- [Attention and KV cache](#attention-and-kv-cache)
- [Model kernels](#model-kernels)
- [Streaming](#streaming)
- [Training](#training)
- [Profiling](#profiling)

## Speculative decoding

| Variable | Meaning |
|----------|---------|
| `GMLX_DSPARK_CONF=T` | Set a DSpark drafter's block-cut confidence threshold, in 0..1. |
| `GMLX_SPEC_GATE=X` | Run a greedy DFlash 2 round expected to emit fewer than X tokens as one plain step, with the same output. `auto` derives X from round costs. |
| `GMLX_QWEN_OWNED=0` | Build Qwen3.5 and 3.6 text MTP targets on stock mlx-vlm classes, without the gmlx performance patches and fixes. |
| `GMLX_GEMMA_OWNED=0` | Build gemma-4 text MTP targets on stock mlx-vlm classes. Numerics are unchanged. |
| `GMLX_GDN_REPLAY=0` | Store the recurrent state after each verify position instead of records that the next step replays. Output is bit-identical. |
| `GMLX_MTP_DEBUG=1` | Log the MTP notices, including the verify branch the first time each branch runs. |

## Prompt cache

The switches for each prompt cache layer are read in `gmlx/spec/engine.py`
and `gmlx/cache/`, and
[Prompt cache internals](prompt-cache.md#the-cache-layers) describes the
layers.

| Variable | Meaning |
|----------|---------|
| `GMLX_SPEC_APC=0` | Turn every speculative cache layer off at once, including lookups, stores, the sidecar and the checkpoint tier. |
| `GMLX_SPEC_APC_RETIRE=0` | Turn off only the retirement store. |
| `GMLX_SPEC_APC_SIDECAR=0` | Turn off only the drafter-KV sidecar. |
| `GMLX_SPEC_APC_CKPT=0` | Turn off only the checkpoint tier, so the exact full-clone path serves those models. |
| `GMLX_SPEC_APC_ENTRIES` | Keep this many entries in the prefix layer. |
| `GMLX_SPEC_APC_SIDECAR_ENTRIES` | Keep this many entries in the drafter sidecar. |
| `GMLX_SPEC_APC_BUDGET_MB` | Keep the prefix layer within this many MB. |
| `GMLX_SPEC_APC_SIDECAR_BUDGET_MB` | Keep the drafter sidecar within this many MB. |
| `GMLX_APC_STORE_EVAL_CHUNK` | Evaluate this many blocks per step in the post-prefill store, which bounds the stall on a long prompt. |
| `GMLX_APC_CKPT_INTERVAL` | Space prefill checkpoints this many tokens apart. `0` saves only the final checkpoint. |
| `GMLX_APC_CKPT_REPLAY=0` | Disable the replay checkpoint, so an identical resend prefills cold again. |
| `GMLX_APC_CKPT_REPLAY_MIN` | Take a replay checkpoint on a recurrent model only for a prompt of at least this many tokens. |
| `GMLX_APC_CKPT_TURN=0` | Disable the turn checkpoint, so next-turn reuse falls back to the interval grid. |
| `GMLX_APC_CKPT_SYS=0` | Disable the system-prompt anchor on both tiers. |
| `GMLX_APC_CKPT_SYS_MIN` | Anchor a system prompt only past this many tokens of shared prefix. |
| `GMLX_APC_ANCHOR_ENTRIES` | Keep this many system-prompt anchors on an exact-tier model. |
| `GMLX_APC_ANCHOR_BUDGET_MB` | Keep the exact-tier anchors within this many MB. The newest anchor is never evicted. |
| `GMLX_APC_CKPT_TRIPWIRE` | Warn about a model that stores nothing, or that matches but adopts nothing, after this many requests. `0` silences both warnings. |
| `GMLX_APC_CKPT_RECORDS` | Keep this many checkpoint records. |
| `GMLX_APC_CKPT_BUDGET_MB` | Keep the checkpoint payload within this many MB. |
| `GMLX_APC_DECODE_CKPT` | Snapshot a checkpoint-tier model at this interval in generated tokens. `0` turns it off. |
| `GMLX_APC_RETIRE_LCP=0` | Key retirement on the forwarded ids instead of the predicted next-turn render. This also disables decode-time snapshots. |
| `GMLX_APC_FRESH_WAIT_MS` | Admit siblings that arrive together one at a time, holding each at most this many ms. `0` disables the gate. |
| `GMLX_APC_FRESH_MIN` | Hold a sibling only past this many uncovered shared-prefix tokens. |

## Attention and KV cache

These switches change how attention runs or how the KV cache is stored, or
log the attention route.

| Variable | Meaning |
|----------|---------|
| `GMLX_KVARN=0` | Disable `--kv-quant-scheme kvarn` at cache build, so the model runs fp16 KV. |
| `GMLX_KVARN_SDPA=0` | Route kvarn decode through the materialize path instead of the fused record kernels. Output differs at fp16 rounding only. |
| `GMLX_KVARN_FA=0` | Keep kvarn MTP verify on the vector kernel instead of the matrix-unit kernel. Numerics match to fp16 rounding. |
| `GMLX_ROPE_FACTORS=0` | Disable the patch that rebuilds Llama-3.1-style per-dimension rope scaling from GGUF metadata. |
| `GMLX_SPARSE_ARCHS` | Name extra architecture modules for the sparse attention route, comma separated, to quality-gate a new architecture. |
| `GMLX_SDPA_DEBUG=1` | Log the attention route of the first long calls, so a wrong route on a new architecture shows in the log. |
| `GMLX_ROUTE_LOG=1` | Print per-route attention call counts at process exit. |

## Model kernels

These switches turn off, resize or trace individual model kernels and
fusions.

| Variable | Meaning |
|----------|---------|
| `GMLX_FUSED_GDN=0` | Disable the fused gated-delta Metal kernels of the Qwen3.5 and 3.6 hybrids. The fusion affects numerics. |
| `GMLX_HADAMARD_KERNEL=0` | Run the Hadamard-fold rotation as MLX ops instead of the mlx-kquant kernel. Numerics match to one rounding. |
| `GMLX_HADAMARD_FUSE=0` | Rotate the input of a folded down or output projection in its own dispatch instead of inside the gate kernel. |
| `GMLX_HADAMARD_TRACE=1` | Count the rotations on a folded file since the last `hadamard_modules.reset_rotation_count` call, read back through `hadamard_modules.rotation_count`. |
| `GMLX_HADAMARD_ROTATE=0` | Skip the Hadamard-fold rotation. The model produces garbage, and a benchmark measures the whole cost of the rotation. |
| `GMLX_MOE_GATEUP_CONCAT=0` | Disable the prefill gate and up expert concat, which trades a second resident copy of the wire bytes for one gather. |
| `GMLX_MOE_GATEUP_CONCAT_MAX_MB` | Cap the concat copies at this many MB, stamping layers in order until the cap is reached. |
| `GMLX_MOE_GATEUP_CONCAT_HEADROOM_GB` | Skip a concat copy that would leave less than this many GB under the memory ceiling. `0` turns the check off. |
| `GMLX_MOE_MIX_PREFILL=0` | Keep the eager unsort and score mix after the sorted-prefill down gather instead of one `gather_mix` dispatch. |
| `GMLX_GLM5_ABSORBED_MAX_L` | Run GLM-5.3-Flash MLA layers in the absorbed MQA form up to this many queries. `0` expands the latent per head everywhere. |
| `GMLX_GLM5_INDEXER_DECODE=0` | Score the GLM-5.3-Flash indexer inline instead of through the fused scorer and radix top-k. |
| `GMLX_GLM5_SPARSE_INDEXED=0` | Disable index-gathered attention for GLM-5.3-Flash sparse decode and verify. |
| `GMLX_GLM5_KDA_FUSED_MAX_T` | Run GLM-5.3-Flash KDA steps up to this width as one fused decode dispatch. `0` restores the op chain everywhere. |
| `GMLX_GLM5_KDA_CHUNK=0` | Keep GLM-5.3-Flash KDA prefill on the token-sequential kernel instead of the chunked recurrence. |
| `GMLX_GLM5_KDA_CONV=0` | Keep the eager GLM-5.3-Flash KDA prefill chain instead of the mlx-kquant glue kernels. |
| `GMLX_HC_M1_MAX_ROWS` | Run hyper-connected steps up to this many rows on the fused per-row kernels. Wider steps take the GEMM route. |
| `GMLX_HC_M1_FUSED=0` | Disable the fused per-row hyper-connection kernels, leaving every step on the GEMM route. |
| `GMLX_HC_FUSED_CYCLE=0` | Restore the two-kernel hyper-connection cycle instead of one mlx-kquant dispatch. Output is bit-identical. |
| `GMLX_DS41_HC_FUSED=0` | Run DeepSeek-V4.1 hyper-connection cycles op by op instead of on the fused kernels. |
| `GMLX_DS41_QAT_FUSED=0` | Run DeepSeek-V4.1 quantization round trips as compiled op chains instead of one kernel each. Output is bit-identical. |
| `GMLX_DS41_SPARSE_KERNEL=0` | Run DeepSeek-V4.1 decode attention as a gather and op chain instead of `sdpa_sparse_decode`. |
| `GMLX_DS41_SPARSE_KERNEL_BLOCK=N` | Set the queries per `sdpa_sparse_decode` call in a DeepSeek-V4.1 prefill. `0` keeps prefill blocks on the op chain. |
| `GMLX_DS41_SPARSE_PREFILL=0` | Run DeepSeek-V4.1 prefill attention as `sdpa_sparse_decode` query blocks instead of one `sdpa_sparse_prefill` call per layer. |
| `GMLX_DS41_INDEXER_DECODE=0` | Compute DeepSeek-V4.1 decode indexer scores on the inline op chain instead of the fused kernel. |
| `GMLX_DS41_POOL_FP4=0` | Keep the DeepSeek-V4.1 latent pool as fp16 rows instead of FP4 codes. Output is bit-identical. |
| `GMLX_DS41_INDEXER_CAND=0` | Score every pooled row under a mask in the later DeepSeek-V4.1 indexer layers instead of only the candidate rows. |
| `GMLX_DS41_PREFILL_STEP=N` | Prefill DeepSeek-V4.1 in N-token chunks. `0` restores the generic default. |
| `GMLX_DS41_PREFILL_TAIL=0` | Run every DeepSeek-V4.1 layer on every prompt row instead of skipping rows no later window reaches. |
| `GMLX_DS4_PREFILL_BLOCK=N` | Set the DeepSeek-V4.1 prefill query-block width for the window, sparse and indexer scores. `0` scores the whole chunk. |
| `GMLX_DSA_INDEXER=0` | Compute DeepSeek-V4 and V4.1 indexer scores and top-k on the inline fp32 op chain instead of the mlx-kquant kernels. |
| `GMLX_DSA_INDEXER_Q=0` | Keep the indexer GEMM on fp16 operands where tensor-op hardware would run the int8 kernel. Output is bit-identical. |
| `GMLX_CB_PHASE=0` | Disable the per-phase MLX command-buffer caps. Output is unchanged, and decode runs slower. |

## Streaming

| Variable | Meaning |
|----------|---------|
| `GMLX_DECODE_LOOKAHEAD_PROBE=1` | Record predicted against actual expert routing per layer and print the recall table at exit, issuing no reads. |
| `GMLX_PIN_CAST_EXCLUDE=0` | Pin the file bytes of every every-token tensor, including the tensors the loader converts at load. |
| `GMLX_STREAM_PLE_COMPOSE=0` | Keep a streamable lookup table resident when the experts also stream. |
| `GMLX_RELEASE_PAGECACHE=0` | Keep a released over-RAM model's pages in the page cache at exit or unload. |

## Training

| Variable | Meaning |
|----------|---------|
| `GMLX_TRAIN_BLOCKED_ATTN=0` | Run training attention on the unfused MLX path instead of the query-block recompute in `gmlx.tune.attention`. |
| `GMLX_TRAIN_GDN_CHUNK=0` | Run a training forward's gated delta scan on mlx-lm's per-token loop instead of the chunked rule in `gmlx.tune.gdn`. |
| `MLX_ENABLE_TF32=1` | Keep TF32 float32 matmul in `gmlx train` and `gmlx distill`, which otherwise turn it off. The chunked gated delta rule then takes the loop. |

## Profiling

These switches time a run's parts and print or write the results.

| Variable | Meaning |
|----------|---------|
| `GMLX_ROUND_PROFILE=1` | Profile each speculative round, in the server process too. `GMLX_ROUND_LOG=/path.tsv` writes the rounds to a TSV file. |
| `GMLX_DECODE_PHASE_STATS=1` | Print the per-token split of a streamed decode between disk stalls and the eval and sync bucket at exit. A clock drop shows as a large sync bucket. |
| `GMLX_DECODE_PHASE_LAYERS=1` | With the phase stats, also print the split per layer and each token's arena misses. |
| `GMLX_DECODE_LAYER_PROFILE=1` | On DeepSeek-V4.1 decode, evaluate after each layer part and print each part's wall time per token. |
| `GMLX_DECODE_LAYER_PROFILE=2` | Also evaluate inside attention and the MoE, so each sub-step is one command buffer in a GPU trace. |
| `GMLX_DECODE_LAYER_PROFILE_LOG` | With the layer profile, write every mark to this path at exit, for aligning a Metal System Trace. |
| `GMLX_LAYER_PROFILE_PREFILL=1` | With the layer profile, also mark the prefill chunks. The per-token figures then average over forward calls. |
