# Hadamard-folded GGUFs

A Hadamard-folded GGUF stores its weights under a rotation, as the PrismML
Ternary Bonsai files do, and gmlx undoes the rotation at run time. Users
need only
[A Hadamard-folded file refuses to load](../troubleshooting.md#a-hadamard-folded-file-refuses-to-load).

## The fold

The rotation spreads each row's outliers before quantization, which
lets a ternary codec hold the model. At run time each folded projection
rotates its input the same way before the matmul, so the product matches
the unfolded model. The header keys under `prism.hadamard.` describe the
fold. `gmlx/load/hadamard.py` parses and checks them, and its docstring
gives the math and the key contract.

## Where the rotation runs

`resolve_hadamard_targets` maps each folded tensor name to its module
path through the loader's name mapping, so the fold follows a tensor
wherever the architecture places it. `install_hadamard_modules` in
`gmlx/load/hadamard_modules.py` then swaps each named `KQuantLinear` and
`KQuantEmbedding` onto a subclass that rotates before the matmul, or after
the gather for an embedding.

The module is the one place every route passes through, so no layer code
needs to know about the fold. The two paths that bypass module calls, the
occupancy fuse and table streaming, refuse a folded module. The rotation
runs on an mlx-kquant kernel when one fits the block width, and as MLX ops
otherwise.

## Sharing one rotation

Projections that read the same activation take one rotation between them
through `shared_linears`, such as q, k and v, or gate and up.
`gmlx.upstream.hadamard_share` swaps the stock attention and MLP onto
forwards that share it. A down or output projection reads a gated
activation, and `glu_rotate` computes that activation and its rotation in
one kernel. A training forward keeps to the MLX ops, because the kernels
have no backward.

## Precision and refusal

A folded file runs at the same activation dtype as any other file.
`--dtype float16` keeps more mantissa bits on the rotated rows and moves
the logprobs closer to the reference, at some cost in speed, so use it for
parity work.

Preflight refuses a folded file whose fold version or architecture the
loader does not support, before any tensor is read, so a fold that gmlx
cannot apply never runs unrotated. The `GMLX_HADAMARD_*` rows of
[Debug switches](debug-switches.md#model-kernels) isolate each part of the
rotation.
