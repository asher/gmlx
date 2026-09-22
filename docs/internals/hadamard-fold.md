# Hadamard-folded GGUFs

How gmlx runs a GGUF whose weights were stored under a Hadamard rotation,
the PrismML Ternary Bonsai files being the first. For contributors who
touch the loader, the qwen35 projection paths or the rotation kernels.
Users need only [troubleshooting.md](../troubleshooting.md) and the
`validate` line that names a folded file.

## The fold

A Hadamard-folded file stores each quantized projection weight as
`W_stored = W H D`, where `D` is a diagonal of signs and `H` is the
normalized natural-order Walsh-Hadamard matrix applied to each contiguous
block of the input dimension. The rotation spreads outliers across the
row before quantization, which is what lets a ternary codec hold the
model. At run time the activation takes the same rotation before the
matmul, so `W_stored (H D x) = W x`. The rotation is its own inverse, and
the file's `token_embd.weight` carries the inverse fold: the gathered row
is transformed and then signed.

The header keys live under `prism.hadamard.` and `gmlx.load.hadamard`
parses them with the checks the reference loader applies: version 1, a
power-of-two block, the fixed transform and axis strings, one sign vector
per input width, `weight_names` for the folded tensors and
`inverse_weight_names` for the embedding. `gdn_v_grouped` means the
`ssm_out` input is first permuted from tiled to grouped value-head order,
`(rep, n_k, head)` from `(n_k, rep, head)`, with the counts from the
`ssm.time_step_rank` and `ssm.group_count` keys. Sign before transform on
the forward fold, transform before sign on the inverse.

## Where the rotation runs

`resolve_hadamard_targets` maps each folded tensor name onto its module
path through the same name mapping the loader uses, so the fold follows a
tensor wherever the arch places it. `install_hadamard_modules` then swaps
the class of every named `KQuantLinear` and `KQuantEmbedding` onto a
subclass whose forward rotates before the matmul (or un-rotates after the
gather), and attaches the fold outside the parameter tree. The module is
the one place every route passes through, including the fused GDN decode
body and the drafter binders that call `embed_tokens` and `lm_head`
directly, so no layer code needs to know about the fold.

The rotation is `kq.hadamard_rotate` when the installed mlx-kquant has it
and the GPU is the default device, and MLX ops otherwise (an f32 upcast,
the sign multiply and `mx.hadamard_transform` per block), which is also
the CPU path. Both forms round once, to the activation dtype.

Two paths bypass module calls and are guarded. The occupancy fuse skips
any projection group that carries a fold, and table streaming refuses a
folded embedding. A LoRA delta multiplies the unrotated input, as the
reference does.

## Sharing one rotation

Projections that read the same activation take one rotation between
them. `shared_linears` rotates once for the members of a group that share
a width, block and sign vector and have no permute, and calls the rest
plainly. It is wired at every decode site: the fused GDN decode and
verify bodies share `in_proj_qkv` and `in_proj_z`, the owned tree's
`verify_linears` shares whatever group it is handed, and
`gmlx.upstream.hadamard_share` swaps the stock mlx-lm attention and MLP
onto forwards that share q/k/v and gate/up. On the 27B this is 258
rotations per decoded token instead of 402. The stock GDN prefill body
still rotates `in_proj_qkv` and `in_proj_z` separately.

The down and output projections read a gated activation, the swiglu and
the attention output gate. Where the installed mlx-kquant has
`glu_hadamard`, `glu_rotate` computes the activation and its rotation in
one kernel and offers the rotated row, and the projection's own rotation
returns that row without a dispatch. The offer matches the array object
and the fold, so any other row still rotates. The stock forwards and the
owned tree both route the activation through `glu_rotate`, which covers
80 of the 258 rotations on the 27B.

## Precision

A folded file runs at the activation dtype every other file gets, bf16
on GPUs with native bf16. The rotation spreads each row's outliers
evenly, so the rotated row has a small dynamic range and the mantissa is
what limits precision on it. float16 keeps three more mantissa bits than
bf16, and `GMLX_ACTIVATION_DTYPE=float16` tightens the teacher-forced
logprob delta against the reference on the 27B from about 0.3 nats to
about 0.04. The bf16 delta already sits inside gmlx's own
prefill-versus-decode noise, and float16 costs speed on a GPU with native
bf16: on the PQ2_0 file an M3 Max decodes about 2 percent slower and the
prefill tile runs about 20 percent slower. The 16K decode integrity test
passes the PQ2_0 file on float16, so the narrower exponent range holds at
depth, and float16 stays the option for parity work rather than the
default. Passing f32 activations changes nothing, because the kquant
matmul's internal precision follows its output dtype.

## Refusal

Preflight refuses a folded file whose header version is not 1 or whose
arch is outside `HADAMARD_ARCHES`, before any tensor is read, so a fold
the loader cannot apply never runs unrotated. An expert stack named as a
fold target raises at resolution for the same reason.

## Switches

`GMLX_HADAMARD_KERNEL`, `GMLX_HADAMARD_FUSE`, `GMLX_HADAMARD_TRACE` and
`GMLX_HADAMARD_ROTATE` are documented in
[debug-switches.md](debug-switches.md).
