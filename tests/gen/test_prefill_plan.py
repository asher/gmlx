"""prefill_plan: the head sub-chunk tiers from the logits cap, the
arithmetic-only plan without a model, and a finite (trunk, step) pair for a
resident and for a streamed teacher through the gmlx seams, monkeypatched."""
from __future__ import annotations

from types import SimpleNamespace

import gmlx.gen.prefill_plan as pp


def test_head_step_takes_the_largest_tier_under_the_cap():
    assert pp.head_step(262144, 4.0, 16) == 512      # 4e9 / (262144 * 16) = 953
    assert pp.head_step(151936, 4.0, 16) == 1024
    assert pp.head_step(262144, 4.0, 20) == 512
    assert pp.head_step(262144, 0.0001, 16) == 1


def test_plan_without_a_model_is_the_arithmetic():
    plan = pp.prefill_plan(None, V=262144)
    assert plan["step"] == 512 and plan["trunk"] == 512 and plan["streaming"] is False
    plan = pp.prefill_plan(None, V=262144, streaming=True, floor=True)
    assert plan["trunk"] == 8192 and plan["bytes_per_v"] == 20
    assert pp.prefill_plan(None, V=1000, requested_trunk=2048)["trunk"] == 2048


def test_plan_with_a_model_returns_finite_pairs(monkeypatch):
    import gmlx.gen.prefill_decay as decay
    import gmlx.stream.expert_streaming as es

    monkeypatch.setattr(es, "_resolve_prefill_step", lambda model, req: (1024, "test"))
    monkeypatch.setattr(decay, "headroom_bytes", lambda: 8e9)
    monkeypatch.setattr(decay, "decayed_step", lambda step, offset, heads: 2048)
    model = SimpleNamespace(args={"text_config": {"num_attention_heads": 16}})

    monkeypatch.setattr(es, "moe_streaming_active", lambda model: False)
    dense = pp.prefill_plan(model, V=151936)
    assert (dense["trunk"], dense["step"], dense["streaming"]) == (512, 1024, False)
    assert dense["trunk_base"] == 1024 and dense["headroom_gb"] == 8.0

    monkeypatch.setattr(es, "moe_streaming_active", lambda model: True)
    streamed = pp.prefill_plan(model, V=151936)
    assert (streamed["trunk"], streamed["streaming"]) == (1024, True)   # capped by the base step
    wide = pp.prefill_plan(model, V=151936, streaming=True, requested_trunk=4096)
    assert wide["trunk"] == 1024
