"""kvarn CLI/config plumbing, eligibility reasons, and the n-aware chat
trim contract. CPU-safe except where marked."""

from __future__ import annotations

import argparse

import pytest


from gmlx.cache.kvarn_cache import kvarn_unsupported, kvarn_widths
from gmlx.gen.generation import setup_kvarn_cache
from kvarn_testlib import FakeLM, needs_kvarn_ops

# -- CLI ---------------------------------------------------------------------


def _parser():
    from gmlx.commands.cli import add_kv_cache_args

    ap = argparse.ArgumentParser()
    add_kv_cache_args(ap)
    return ap


def test_cli_accepts_kvarn_flags():
    args = _parser().parse_args(
        ["--kv-quant-scheme", "kvarn", "--kv-tail-tokens", "256"]
    )
    assert args.kv_quant_scheme == "kvarn"
    assert args.kv_tail_tokens == 256


def test_cli_defaults():
    args = _parser().parse_args([])
    assert args.kv_quant_scheme is None
    assert args.kv_tail_tokens == 1024


def test_cli_rejects_unknown_scheme():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--kv-quant-scheme", "affine"])


def test_config_keys_registered():
    from gmlx.commands.cli import _CFG_LOAD_TO_ARG
    from gmlx.config import LOAD_ENV

    assert LOAD_ENV["kv_quant_scheme"] == "KV_QUANT_SCHEME"
    assert LOAD_ENV["kv_tail_tokens"] == "KV_TAIL_TOKENS"
    assert _CFG_LOAD_TO_ARG["kv_quant_scheme"] == "kv_quant_scheme"
    assert _CFG_LOAD_TO_ARG["kv_tail_tokens"] == "kv_tail_tokens"


# -- width selection ---------------------------------------------------------


def test_kvarn_widths(monkeypatch):
    monkeypatch.delenv("GMLX_KVARN_BITS", raising=False)
    assert kvarn_widths(None) == (6, 6)
    assert kvarn_widths(4) == (4, 4)
    monkeypatch.setenv("GMLX_KVARN_BITS", "k6v5")
    assert kvarn_widths(None) == (6, 5)
    monkeypatch.setenv("GMLX_KVARN_BITS", "bogus")
    assert kvarn_widths(8) == (8, 8)


# -- eligibility -------------------------------------------------------------


def test_unsupported_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_KVARN", "0")
    assert "GMLX_KVARN=0" in kvarn_unsupported(FakeLM())


def test_unsupported_reasons(kvarn_ops_ok, monkeypatch):
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    assert kvarn_unsupported(FakeLM()) is None
    assert kvarn_unsupported(FakeLM(head_dim=256)) is None
    assert kvarn_unsupported(FakeLM(head_dim=512)) is None
    assert "head_dim 64" in kvarn_unsupported(FakeLM(head_dim=64))
    assert "MLA" in kvarn_unsupported(FakeLM(kv_lora_rank=512))
    # qwen3.5 lost its bypass entry when the owned dispatch gained the arm
    assert kvarn_unsupported(FakeLM(model_type="qwen3_5")) is None


def test_unsupported_mixed_dims(kvarn_ops_ok, monkeypatch):
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    # gemma-4 shape: sliding head_dim 256 + global_head_dim 512 on the
    # convertible layers; any supported dim passes the gate.
    m = FakeLM(head_dim=256)
    m.args.global_head_dim = 512
    assert kvarn_unsupported(m) is None
    m64 = FakeLM(head_dim=64)
    m64.args.global_head_dim = 96
    reason = kvarn_unsupported(m64)
    assert "head_dim 64/96" in reason and "128/256/512" in reason


def test_unsupported_gemma4_owned_tree(kvarn_ops_ok, monkeypatch):
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    import gmlx.models.gemma4.owned as gemma4_owned

    owned = FakeLM(head_dim=256)
    monkeypatch.setattr(gemma4_owned, "is_owned_language_model", lambda m: m is owned)
    assert "owned (MTP) tree" in kvarn_unsupported(owned)
    assert kvarn_unsupported(FakeLM(head_dim=256)) is None
    # serve wraps the target; the owned tree is the language model inside
    from types import SimpleNamespace

    wrapper = SimpleNamespace(language_model=owned, config={})
    assert "owned (MTP) tree" in kvarn_unsupported(wrapper)


def test_setup_rejects_bad_bits_and_tail(kvarn_ops_ok, monkeypatch, capsys):
    # a width or tail the scheme rejects exits 2, like affine's
    # resolve_and_report; only a model-shape decline runs fp16
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    with pytest.raises(SystemExit) as e:
        setup_kvarn_cache(FakeLM(), 7, 1024, None)
    assert e.value.code == 2
    assert "error: kvarn bits" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        setup_kvarn_cache(FakeLM(), 6, 100, None)
    assert e.value.code == 2
    assert "kv_tail_tokens" in capsys.readouterr().err
    assert setup_kvarn_cache(FakeLM(head_dim=64), 6, 1024, None) is None
    assert "dropped" in capsys.readouterr().err


@needs_kvarn_ops
def test_setup_builds_cache_and_banner(monkeypatch, capsys):
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    pc = setup_kvarn_cache(FakeLM(), None, 1024, None)
    assert pc is not None and len(pc) == 2
    assert all(type(c).__name__ == "KVarNKVCache" for c in pc)
    err = capsys.readouterr().err
    assert "[kv] kvarn6 tail=1024 -> quantized 2/2 attn layers" in err


@needs_kvarn_ops
def test_setup_declines_rotating_stack(monkeypatch, capsys):
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    # max_kv_size forces rotating caches; kvarn leaves them fp16 and the
    # scheme drops loudly rather than silently.
    model = FakeLM()

    def rotating_cache():
        from mlx_lm.models.cache import RotatingKVCache

        return [RotatingKVCache(max_size=64) for _ in range(2)]

    model.make_cache = rotating_cache
    assert setup_kvarn_cache(model, None, 1024, 64) is None
    assert "no kvarn-convertible layers" in capsys.readouterr().err


# -- chat n-aware trim -------------------------------------------------------


class _TrimKV:
    def __init__(self, offset=10, ret="n"):
        self.offset = offset
        self._ret = ret

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        if self._ret == "none":
            return None
        if self._ret == "short":
            return n - 1
        return n


class _ProbeKV(_TrimKV):
    def __init__(self, offset=10, ok_below=5):
        super().__init__(offset)
        self.ok_below = ok_below

    def _can_trim(self, n):
        return n <= self.ok_below


def test_chat_can_trim_prefers_probe():
    from gmlx.tui.chat import _can_trim

    assert _can_trim([_TrimKV(), _ProbeKV(ok_below=5)], 5)
    assert not _can_trim([_TrimKV(), _ProbeKV(ok_below=5)], 6)
    # n-blind form falls back to is_trimmable
    assert _can_trim([_TrimKV(), _ProbeKV(ok_below=0)])


def test_chat_trim_to_checks_returns():
    from gmlx.tui.chat import _trim_to

    ok = [_TrimKV(10), _TrimKV(10)]
    assert _trim_to(ok, 4)
    assert all(c.offset == 4 for c in ok)
    legacy = [_TrimKV(10, ret="none")]
    assert _trim_to(legacy, 4)
    lying = [_TrimKV(10, ret="short")]
    assert not _trim_to(lying, 4)


def test_chat_trim_to_respects_probe():
    from gmlx.tui.chat import _trim_to

    cache = [_ProbeKV(offset=10, ok_below=3)]
    assert not _trim_to(cache, 4)  # needs 6, probe caps at 3
    assert cache[0].offset == 10  # untouched
    assert _trim_to(cache, 8)  # needs 2


# -- affine-path hygiene -----------------------------------------------------


@needs_kvarn_ops
def test_kvarn_caches_dodge_affine_probes(monkeypatch):
    # No ``bits`` attr (would route hasattr-based quantized SDPA) and no
    # ``to_quantized`` (would let maybe_quantize_kv_cache convert mid
    # stream); the kvarn arm nulls kv_bits so neither probe ever matters.
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    pc = setup_kvarn_cache(FakeLM(), None, 1024, None)
    assert pc is not None
    assert all(not hasattr(c, "bits") for c in pc)
    assert all(not hasattr(c, "to_quantized") for c in pc)


def test_head_dims_flag_every_mla_arch(kvarn_ops_ok, monkeypatch):
    """MLA stacks store one latent per layer; kvarn declines them by shape.
    deepseek_v4 has no kv_lora_rank field, so its compress_ratios must
    carry the marker, or its head_dim 512 latent would pass as supported."""
    from types import SimpleNamespace

    from gmlx.cache.kvarn_cache import kvarn_head_dims
    from gmlx.models.deepseek_v4.model import ModelArgs as Ds4Args
    from gmlx.models.glm5_next.model import ModelArgs as GlmArgs

    monkeypatch.delenv("GMLX_KVARN", raising=False)
    ds4 = SimpleNamespace(args=Ds4Args())
    glm = SimpleNamespace(args=GlmArgs(
        model_type="glm5_next", vocab_size=8, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=64, rms_norm_eps=1e-5))
    for lm in (ds4, glm):
        assert kvarn_head_dims(lm) == {-1}
        assert "MLA" in kvarn_unsupported(lm)
    plain = SimpleNamespace(args=SimpleNamespace(head_dim=512))
    assert kvarn_head_dims(plain) == {512}
    assert kvarn_unsupported(plain) is None


def test_resolve_kwargs_is_the_one_assembly_point(kvarn_ops_ok, monkeypatch):
    from gmlx.cache.kvarn_cache import kvarn_resolve_kwargs

    kw = kvarn_resolve_kwargs(FakeLM())
    assert kw == dict(scheme="kvarn", kv_bits=6, value_bits=6, tail_tokens=1024,
                      rotating_window=None, scheme_reason=None)
    kw = kvarn_resolve_kwargs(FakeLM(), 4, 3, 256, 4096)
    assert (kw["kv_bits"], kw["value_bits"], kw["tail_tokens"],
            kw["rotating_window"]) == (4, 3, 256, 4096)
    assert kvarn_resolve_kwargs(FakeLM(), 4, None, 0)["tail_tokens"] == 0
    # the env split applies only when no value width is given
    monkeypatch.setenv("GMLX_KVARN_BITS", "k6v5")
    assert kvarn_resolve_kwargs(FakeLM())["value_bits"] == 5
    assert kvarn_resolve_kwargs(FakeLM(), value_bits=4)["value_bits"] == 4
    assert "MLA" in kvarn_resolve_kwargs(FakeLM(kv_lora_rank=8))["scheme_reason"]
