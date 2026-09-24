"""Training-time helpers: attention and recurrent scans with bounded
memory, per-layer activation checkpointing, and the LoRA student setup
that ``gmlx train`` and ``gmlx distill train`` share. Inference paths do
not import this package."""
