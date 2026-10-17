# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: the defaults of the
# flux.2-klein-9b-kv row of FLUX2_MODEL_INFO in src/flux2/util.py, and
# modified for gmlx (Apache-2.0; see licenses/flux2-LICENSE).
"""FLUX.2 klein 9B KV registry row.

KV is a fine-tune of FLUX.2 klein 9B for edits. Its reference tokens
attend only to each other, at time zero, so their keys and values in
every block do not change from step to step. A conditioning runs them
through the blocks once, and every step reads their keys and values
after its own. Text to image runs the plain forward.

KV has the names and shapes of the 9B files, the 9B encoder, schedule,
sampling loop and reference image rules, and other weights. This row
comes before the 9B row and takes a 9B file whose image input
projection or KV marker has KV's hash.
"""

from __future__ import annotations

import dataclasses

from gmlx.image.families.flux2_klein import family as klein
from gmlx.image.families.flux2_klein_9b import family as nine
from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow, Family, StepParts, VariantRow

TITLE = nine.KV_TITLE

# The official transformer config of KV gives the 9B constants.
CONSTANTS = ConstantsRow(
    values=klein.CONSTANTS.values,
    repo="black-forest-labs/FLUX.2-klein-9b-kv",
    revision="a6dfb36eca3a3906eb2fd460795adfb844e5fcce",
)

VARIANTS = {
    nine.KV_HASH: VariantRow(
        "distilled", "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        "f4b7b4b0750b7b5cd112f9cc6e386a2ded10061f",
        tuple(f"Flux-2-Klein-9B-KV-{q}.gguf"
              for q in ("Q2_K", "Q4_K_M", "Q8_0"))),
}

# The defaults of flux.2-klein-9b-kv, 4 steps and guidance 1, are those of
# the distilled 9B.
SAMPLING = {"distilled": klein.SAMPLING["distilled"]}
# KV has no checkpoint that takes guidance.
UNGUIDED = {"distilled": ""}

# The states of other conditionings a pipeline holds beside those of the
# request it runs. One 1024 x 1024 reference holds 2 GiB.
STATE_BYTES = 1 << 30


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if klein.flux_model(view) != nine.TITLE or not nine.is_kv(view):
        return None
    return klein.relative_refusal(view, nine.SIZE) or Match()


def _modulation(dit, sigma):
    from .step import modulation

    return modulation(dit, sigma)


def _conditioning(dit, cond):
    from .step import conditioning

    return conditioning(dit, cond)


def _forward(dit, latents, sigma, cond, state, mod):
    from .step import forward

    return forward(dit, latents, sigma, cond, state, mod)


def _state_bytes(dit, cond) -> int:
    from .step import state_bytes

    return state_bytes(dit, cond)


def lora_map(config: dict):
    return dataclasses.replace(klein.lora_map(config),
                               family="flux2_klein_9b_kv", title=TITLE)


FAMILY = Family(
    name="flux2_klein_9b_kv",
    title=TITLE,
    # Before the FLUX.2 klein 9B row, which then reads the plain 9B files.
    order=23,
    arch_strings=klein.FAMILY.arch_strings,
    shared_arch_strings=klein.FAMILY.shared_arch_strings,
    strip_prefixes=klein.FAMILY.strip_prefixes,
    fingerprint=fingerprint,
    synth_config=klein.synth_config,
    constants=CONSTANTS,
    build=klein.build,
    lora_map=lora_map,
    fp32_keep=klein.FAMILY.fp32_keep,
    encoder=nine.encoder,
    identity_tensors=klein.IDENTITY_TENSORS,
    variants=VARIANTS,
    sampling=SAMPLING,
    unguided=UNGUIDED,
    unknown_variant="distilled",
    step=StepParts(_modulation, _conditioning, _forward, STATE_BYTES,
                   _state_bytes),
    schedule=klein.schedule,
    loop=klein.LOOP,
    edit=klein.FAMILY.edit,
)
