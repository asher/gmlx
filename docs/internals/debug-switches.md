# Debug switches

Environment variables that change how gmlx builds or routes a model, or
make it report what it did, so that a defect can be isolated. None is a
tuning setting. The disabling switches slow the model or turn a fix off,
the logging switches cost nothing but output, and all of them exist so that
an A/B run can rule a component in or out. They are read at load or on each
call. The user-facing variables are in [env-vars.md](../env-vars.md).

| Variable | Meaning |
|----------|---------|
| `GMLX_KVARN=0` | Disable `--kv-quant-scheme kvarn` at cache build. The scheme is dropped with that reason and the model runs fp16 KV. |
| `GMLX_KVARN_SDPA=0` | Route kvarn decode through the materialize path instead of the fused record kernels. Differs at fp16 rounding only. Set this first when debugging kvarn. |
| `GMLX_KVARN_FA=0` | Keep kvarn MTP verify rounds on the vector decode kernel instead of the matrix-unit kernel. Same numerics to fp16 rounding. Widths above 4 materialize. |
| `GMLX_DECODE_LOOKAHEAD_PROBE=1` | Record predicted versus actual expert routing per layer and print the recall table at exit, issuing no reads. Run it on a new model family. |
| `GMLX_ROPE_FACTORS=0` | Disable the patch that rebuilds Llama-3.1-style per-dimension rope scaling from GGUF metadata. Set only to rule it out when debugging long context. |
| `GMLX_SPARSE_ARCHS` | Extra architecture modules the sparse attention route may apply to, comma separated, for a quality gate on a new arch. The default is the llama family only. |
| `GMLX_FUSED_GDN=0` | Disable the fused gated-delta Metal kernels the Qwen3.5 and 3.6 hybrids use. The fusion affects numerics, so set this first when debugging those archs. |
| `GMLX_QWEN_OWNED=0` | Build Qwen3.5 and 3.6 text MTP targets on stock mlx-vlm classes. Disables all performance patches and restores two stock defects. Multimodal targets ignore it. |
| `GMLX_GEMMA_OWNED=0` | Build gemma-4 text MTP targets on stock mlx-vlm classes. Numerics are unchanged either way. Multimodal targets ignore it. |
| `GMLX_MOE_GATEUP_CONCAT=0` | Disable the prefill gate and up expert concat, which runs one gather over the concatenated wire bytes at the cost of a second resident copy of them. |
| `GMLX_MOE_GATEUP_CONCAT_MAX_MB` | Cap in MB on the concat copies the install builds, default `2048`. Layers are stamped in order until the cap is reached. |
| `GMLX_MOE_GATEUP_CONCAT_HEADROOM_GB` | Room left under the memory ceiling after a concat copy, default `8`. A copy that would not fit is skipped and logged. `0` turns the check off. |
| `GMLX_MOE_MIX_PREFILL=0` | Keep the eager unsort and score mix after the sorted-prefill MoE down gather instead of running them as one mlx-kquant `gather_mix` dispatch. |
| `GMLX_GLM5_ABSORBED_MAX_L` | Widest query count GLM-5.3-Flash MLA layers run in the absorbed MQA form instead of expanding the latent per head, default `16`. `0` restores the expansion. |
| `GMLX_GLM5_INDEXER_DECODE=0` | Score the GLM-5.3-Flash DSA indexer inline instead of through the mlx-kquant fused scorer and radix top-k. The fused route rounds scores before the select. |
| `GMLX_GLM5_SPARSE_INDEXED=0` | Disable index-gathered attention for GLM-5.3-Flash sparse decode and verify, restoring the per-query gather and sdpa loop. The route needs `sdpa_fa_indexed`. |
| `GMLX_GLM5_KDA_FUSED_MAX_T` | Widest step GLM-5.3-Flash KDA layers run as one fused decode dispatch per layer, default `8`. Wider steps take the op chain, and `0` restores it everywhere. |
| `GMLX_GLM5_KDA_CHUNK=0` | Keep GLM-5.3-Flash KDA prefill on the token-sequential kernel instead of the mlx-kquant chunked recurrence, which needs tensor-op hardware and head dim 128. |
| `GMLX_GLM5_KDA_CONV=0` | Keep the eager GLM-5.3-Flash KDA prefill chain instead of the mlx-kquant glue `kda_conv`, `kda_chunk_gated` and `rmsnorm_gate`. |
| `GMLX_HC_M1_MAX_ROWS` | Widest step in rows that hyper-connected models run the fused per-row hyper-connection kernels on, default `8`. Wider steps take the GEMM route. |
| `GMLX_HC_M1_FUSED=0` | Disable the fused per-row hyper-connection kernels on DeepSeek-V4 and GLM-5.3-Flash, leaving every step on the GEMM route. |
| `GMLX_HC_FUSED_CYCLE=0` | Restore the two-kernel hyper-connection cycle instead of running the expand, front reduction and collapse as one mlx-kquant dispatch. Bit-identical either way. |
| `GMLX_DS41_HC_FUSED=0` | DeepSeek-V4.1 hyper-connection cycles on the op-by-op chain instead of the fused kernels, at decode and at prefill width. Same numerics to bfloat16 rounding. |
| `GMLX_DS41_QAT_FUSED=0` | DeepSeek-V4.1 window-KV and indexer quantization round-trips as compiled op chains instead of one mlx-kquant kernel each. Bit-identical. |
| `GMLX_DS41_SPARSE_KERNEL=0` | DeepSeek-V4.1 decode attention as the gather and compiled op chain instead of the mlx-kquant `sdpa_sparse_decode` kernel. The kernel keeps its softmax in fp32. |
| `GMLX_DS4_PREFILL_BLOCK=N` | DeepSeek-V4.1 prefill query-block width for the window, sparse and indexer scores, default `512`. `0` scores every query against the whole chunk. |
| `GMLX_DSA_INDEXER=0` | DeepSeek-V4 and V4.1 indexer scores and top-k on the inline fp32 op chain instead of the mlx-kquant GEMM and radix select. Same picks to fp16 rounding. |
| `GMLX_DSA_INDEXER_Q=0` | Keep the indexer GEMM on fp16 operands where tensor-op hardware would run the int8 kernel on the packed FP4 codes. Bit-identical. |
| `GMLX_DS41_SPARSE_KERNEL_BLOCK=N` | Queries per `sdpa_sparse_decode` call in a DeepSeek-V4.1 prefill, default `64`. `0` keeps prefill blocks on the op chain. |
| `GMLX_DS41_SPARSE_PREFILL=0` | DeepSeek-V4.1 prefill attention as `sdpa_sparse_decode` query blocks instead of one `sdpa_sparse_prefill` call per layer. Same numerics. |
| `GMLX_DS41_INDEXER_DECODE=0` | DeepSeek-V4.1 decode-width indexer scores on the inline fp32 op chain instead of the fused mlx-kquant kernel. Same picks to fp16 rounding. |
| `GMLX_DS41_POOL_FP4=0` | Keep the DeepSeek-V4.1 latent pool as fp16 rows at rest instead of the FP4 codes and scales the sparse kernels read directly. Bit-identical. |
| `GMLX_DS41_INDEXER_CAND=0` | DeepSeek-V4.1 decode indexer layers past the candidate source score every pooled row under a mask instead of only the listed candidate rows. Same picks. |
| `GMLX_DS41_PREFILL_STEP=N` | DeepSeek-V4.1 prefills in N-token chunks instead of 8192 when the experts stream and 4096 when they are resident. `0` restores the 2048-token default. |
| `GMLX_DS41_PREFILL_TAIL=0` | Run every DeepSeek-V4.1 layer on every prompt row. By default the layers past the last kv-source layer skip rows no later window reaches. Same logits. |
| `GMLX_CB_PHASE=0` | Disable the per-phase MLX command-buffer caps, fine through a prefill and coarse from the first generated token. Output is unchanged. Decode runs slower. |
| `GMLX_SDPA_DEBUG=1` | Log which attention route each layer took, so a wrong route on a new architecture shows in the log. |
| `GMLX_ROUTE_LOG=1` | Print per-route attention call counts at process exit. |
| `GMLX_MTP_DEBUG=1` | Log the MTP verify branch per round. |
| `GMLX_ROUND_PROFILE=1` | Profile each speculative round, in the server process too, so a serve claim can be certified there. `GMLX_ROUND_LOG=/path.tsv` writes the rounds to a TSV file. |
| `GMLX_DECODE_PHASE_STATS=1` | Print a streamed decode's per-token split between disk stalls and the eval and sync bucket at exit. A clock frequency drop shows as a large sync bucket. |
| `GMLX_DECODE_PHASE_LAYERS=1` | With the phase stats, also print the split per layer and each token's arena misses, so a cold layer or a cold start shows where it is. |
| `GMLX_DECODE_LAYER_PROFILE=1` | DeepSeek-V4.1 decode: eval after each layer component and print the wall per token of attention, experts, hyper-connections and engram at exit. Slows the run. |
| `GMLX_DECODE_LAYER_PROFILE=2` | Also eval inside attention and the MoE, so each sub-step (projections, indexer, core, router, experts, shared expert) is one command buffer in a GPU trace. |
| `GMLX_DECODE_LAYER_PROFILE_LOG` | With the layer profile, write every mark as `key layer wall_t0 wall_t1` to this path at exit, for aligning a Metal System Trace to the marks. |
| `GMLX_LAYER_PROFILE_PREFILL=1` | With the layer profile, also mark the steps wider than one token (the prefill chunks); the per-token figures then average over forward calls. |
| `GMLX_PIN_CAST_EXCLUDE=0` | Pin the file bytes of every every-token tensor on a streamed model, including tensors the loader converts at load. By default converted tensors are left out. |
| `GMLX_STREAM_PLE_COMPOSE=0` | Keep a streamable lookup table resident when the experts also stream, instead of streaming both. |
| `GMLX_RELEASE_PAGECACHE=0` | Keep a released over-RAM model's pages in the page cache at exit or unload instead of invalidating them. |
