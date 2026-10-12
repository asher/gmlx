"""A native MTP head attends at the trunk's positions, so under rope
scaling its ropes carry the target's resolved table and magnitude. Tiny
synthesized Qwen 3.5 and Hy v3 targets; the head is captured as the loader
builds it, before any weights load."""

from __future__ import annotations

import mlx.nn as nn
import numpy as np
import pytest

from gmlx.load.config_synth import synthesize_config
from gmlx.load.loader import build_model
from gmlx.load.rope_scaling import RopeOverride
from test_config_synth import _HY_V3_SHAPES, _hy_v3_meta
from test_qwen35_rope_scaling import _SHAPES as _Q35_SHAPES
from test_qwen35_rope_scaling import _meta as _q35_meta

YARN4 = RopeOverride(rope_scaling="yarn", rope_scale=4.0)


class _Built(Exception):
    pass


def _rope_state(r):
    """(table, magnitude) of a rope module, whatever its class."""
    if hasattr(r, "attention_scaling") and hasattr(r, "inv_freq"):
        return np.asarray(r.inv_freq), float(r.attention_scaling)
    freqs = getattr(r, "_freqs", None)
    if freqs is not None:
        return 1.0 / np.asarray(freqs), float(getattr(r, "mscale", 1.0))
    return None, None


def _ropes(model):
    out = []
    for _, m in model.named_modules():
        for attr in ("rope", "rotary_emb"):
            r = getattr(m, attr, None)
            if isinstance(r, nn.Module):
                out.append(r)
    return out


def _head(monkeypatch, cfg, target, arch):
    import gmlx.spec.mtp_load as mtp_load

    seen = {}

    def install(drafter, *a, **k):
        seen["drafter"] = drafter
        raise _Built

    monkeypatch.setattr(mtp_load, "remap_mtp_arrays",
                        lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(mtp_load, "_install_and_load", install)
    with pytest.raises(_Built):
        mtp_load._load_mtp_drafter({}, {}, arch, cfg, target,
                                   log=lambda *a, **k: None)
    return seen["drafter"]


@pytest.mark.parametrize("arch,meta,shapes", [
    # Four trunk layers (the last one full attention) and one MTP block.
    ("qwen35", lambda: {**_q35_meta(), "qwen35.block_count": 5,
                        "qwen35.nextn_predict_layers": 1}, _Q35_SHAPES),
    ("hy_v3", _hy_v3_meta, _HY_V3_SHAPES),
])
def test_mtp_head_rotates_as_its_target(monkeypatch, arch, meta, shapes):
    cfg = synthesize_config(meta(), dict(shapes), rope_override=YARN4)
    assert int(cfg.get("mtp_num_hidden_layers", 0)) >= 1
    from gmlx.load import rope_swap

    target, cfg = build_model(cfg, mtp=True)
    assert rope_swap.apply(target, cfg) > 0
    want_table, want_mag = _rope_state(_ropes(target)[0])
    assert want_mag > 1.0
    head = _head(monkeypatch, cfg, target, arch)
    ropes = _ropes(head)
    assert ropes
    for r in ropes:
        table, mag = _rope_state(r)
        assert mag == pytest.approx(want_mag)
        np.testing.assert_allclose(table, want_table, rtol=1e-6)
