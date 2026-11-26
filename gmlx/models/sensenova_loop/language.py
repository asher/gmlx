# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""SenseNova Loop text model: the owned Qwen3.5 hybrid with a layer loop
and two-stream mHC.

Reference: llama.cpp ``src/models/sensenova-loop.cpp`` (OpenSenseNova fork,
MIT) and the Hugging Face remote code ``modeling_sensenova_loop.py``.

- Loop: physical layers ``[start, end)`` run ``count`` times with shared
  weights. Every visit keeps its own KV or recurrent state, so the cache
  list, ``LanguageModel.layers`` and the forward walk follow the visits
  (``execution_layers``) while the parameter tree holds each physical
  layer once.
- mHC: layers ``[mhc.start, mhc.end)`` carry two residual streams. The
  first mHC layer of a pass starts both streams from its input, every mHC
  layer reads and writes them through its routers (``mhc.MHCRouter``),
  and the last one hands the next layer their mean. The streams are
  float32 as in llama.cpp (``GMLX_SENSENOVA_STREAMS_F32=0`` keeps them in
  the activation dtype).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.qwen3_5.config import ModelConfig as _Q35ModelConfig
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig as _Q35TextConfig

from gmlx.models.qwen35.layers import OwnedQwen3_5DecoderLayer
from gmlx.models.qwen35.owned import OwnedQwen3_5LanguageModel, OwnedQwen3_5Model

from . import mhc as _mhc

MODEL_TYPE = "sensenova_loop"


@dataclass
class TextConfig(_Q35TextConfig):
    # {"start", "end", "count"}: physical layers [start, end) run count times.
    loop: Optional[Dict[str, int]] = None
    # {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}
    mhc: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        n = self.num_hidden_layers
        loop = self.loop or {"start": 0, "end": n, "count": 1}
        start, end, count = int(loop["start"]), int(loop["end"]), int(loop["count"])
        if not (0 <= start < end <= n and count >= 1):
            raise ValueError(f"sensenova_loop: invalid loop {loop} for {n} layers")
        self.loop = {"start": start, "end": end, "count": count}
        if self.mhc is not None:
            m = self.mhc
            if int(m.get("streams", 2)) != 2:
                raise ValueError("sensenova_loop: only two mHC streams are supported")
            ms, me = int(m["start"]), int(m["end"])
            if not (0 <= ms < me <= n):
                raise ValueError(f"sensenova_loop: invalid mHC range {m}")
            if not (me <= start or (start <= ms and me <= end) or ms >= end):
                raise ValueError("sensenova_loop: the mHC range crosses a loop boundary")


def execution_plan(args) -> list[int]:
    """Physical layer per visit, in execution order."""
    loop = args.loop
    s, e, c = loop["start"], loop["end"], loop["count"]
    return (list(range(s)) + list(range(s, e)) * c
            + list(range(e, args.num_hidden_layers)))


class SenseNovaLoopDecoderLayer(OwnedQwen3_5DecoderLayer):
    """A physical layer inside the mHC range: the Qwen3.5 layer plus one
    router before attention and one before the MLP."""

    def __init__(self, args, layer_idx: int):
        super().__init__(args, layer_idx)
        self.mhc_attn = _mhc.MHCRouter(args.hidden_size, args.mhc)
        self.mhc_ffn = _mhc.MHCRouter(args.hidden_size, args.mhc)

    @property
    def activation_dtype(self):
        return self.input_layernorm.weight.dtype

    def mhc_step(self, state, mask=None, cache=None, position_ids=None,
                 position_embeddings=None, gdn_sink=None, target_verify=False):
        """``state`` is (streams ``[..., 2 * hidden]``, pending branch
        output or None, its coefficients). Returns the state with this
        layer's MLP output pending, so the next router merges it."""
        act = self.activation_dtype
        s, y, coef = state
        s, x, coef = _mhc.route(self.mhc_attn, s, y, coef, act)
        x = self.input_layernorm(x)
        if self.is_linear:
            r = self.linear_attn(x, mask, cache, gdn_sink=gdn_sink,
                                 target_verify=target_verify)
        else:
            r = self.self_attn(x, mask=mask, cache=cache,
                               position_ids=position_ids,
                               position_embeddings=position_embeddings,
                               target_verify=target_verify)
        s, x, coef = _mhc.route(self.mhc_ffn, s, r, coef, act)
        r = self.mlp(self.post_attention_layernorm(x), target_verify)
        return s, r, coef


class LayerVisit:
    """One visit of a physical layer. Attribute reads go to the layer, so
    code that walks ``layers`` beside the cache list sees ``is_linear``,
    ``self_attn`` and ``linear_attn`` as usual. Not a module: the
    parameter tree holds the physical layer once."""

    __slots__ = ("layer", "physical", "enter", "leave", "streams_f32")

    def __init__(self, layer, physical: int, enter: bool, leave: bool,
                 streams_f32: bool = True):
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "physical", physical)
        object.__setattr__(self, "enter", enter)
        object.__setattr__(self, "leave", leave)
        object.__setattr__(self, "streams_f32", streams_f32)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "layer"), name)

    def __call__(self, h, **kwargs):
        layer = self.layer
        if not isinstance(layer, SenseNovaLoopDecoderLayer):
            return layer(h, **kwargs)
        state = (_mhc.enter(h, self.streams_f32), None, None) if self.enter else h
        state = layer.mhc_step(state, **kwargs)
        if not self.leave:
            return state
        s, y, coef = state
        return _mhc.leave(_mhc.merge(s, y, coef), layer.activation_dtype)


class SenseNovaLoopModel(OwnedQwen3_5Model):
    """The owned Qwen3.5 model scaffold over the visit list."""

    def __init__(self, args: TextConfig):
        nn.Module.__init__(self)
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        m = args.mhc

        def in_mhc(i):
            return m is not None and m["start"] <= i < m["end"]

        self.layers = [
            (SenseNovaLoopDecoderLayer if in_mhc(i) else OwnedQwen3_5DecoderLayer)(
                args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1
        streams_f32 = os.environ.get("GMLX_SENSENOVA_STREAMS_F32", "1") != "0"
        visits = []
        for p in execution_plan(args):
            visits.append(LayerVisit(
                self.layers[p], p,
                enter=m is not None and p == m["start"],
                leave=m is not None and p == m["end"] - 1,
                streams_f32=streams_f32))
        # Stored outside the module dict: not part of the parameter tree.
        object.__setattr__(self, "_visits", tuple(visits))
        plan = [v.physical for v in visits]
        for i, v in enumerate(visits):
            if v.is_linear != (not self._full_attention(i)):
                raise ValueError(
                    f"sensenova_loop: visit {i} (layer {plan[i]}) changes the "
                    "attention type of its slot")

    def _full_attention(self, visit: int) -> bool:
        return (visit + 1) % self.args.full_attention_interval == 0

    @property
    def execution_layers(self):
        return self._visits


class SenseNovaLoopLanguageModel(OwnedQwen3_5LanguageModel):
    """The owned Qwen3.5 language model over ``SenseNovaLoopModel``. All
    speculative hooks and the rollback are inherited; they act on the
    visit-ordered cache list."""

    def __init__(self, args: TextConfig, config: _Q35ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = SenseNovaLoopModel(args)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return list(self.model.execution_layers)

    @property
    def cast_predicate(self):
        def predicate(path: str):
            return not (path.endswith("A_log") or ".mhc_" in path)

        return predicate


def model_config(args: TextConfig) -> _Q35ModelConfig:
    """A container config for the text-only builds: mrope reads
    ``vision_config`` even for text, so a default one goes in."""
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import VisionConfig

    return _Q35ModelConfig(
        text_config=args, vision_config=VisionConfig(), model_type=MODEL_TYPE,
        vocab_size=args.vocab_size)
