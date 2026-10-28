#!/usr/bin/env python3
"""`cli.main` run-path flag plumbing: --help-documented flags reach
generation.generate on the plain (non-MTP, non-VLM) path, --prompt-file feeds the
prompt, --report-only inventories without building, and the documented exit
codes (1 = load refusal, 2 = bad invocation) hold. CPU-only: every load seam
is a stub."""

from __future__ import annotations

import pytest

import gmlx.commands.cli as cli


@pytest.fixture(autouse=True)
def no_family_defaults(monkeypatch):
    """Keep argparse defaults: family seeding would fill flags we didn't pass."""
    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def gen(monkeypatch):
    """Stub the plain-path load/generate seams; returns the recorded generate call."""
    import gmlx.gen.generation as generation
    import gmlx.load.loader as loader

    seen = {}

    def fake_generate(model, tok, prompt, **kwargs):
        seen["prompt"] = prompt
        seen.update(kwargs)

    monkeypatch.setattr(loader, "load_model",
                        lambda *a, **k: (object(), {}, object()))
    monkeypatch.setattr(generation, "generate", fake_generate)
    return seen


# flag forwarding on the plain generate path
def test_run_forwards_sampling_and_kv_flags(gguf, gen):
    rc = cli.main([gguf,
                   "--stop", "END", "--stop", "DONE",
                   "--kv-bits", "8", "--kv-group-size", "32",
                   "--max-kv-size", "4096", "--quantized-kv-start", "16",
                   "--thinking-budget", "100",
                   "--repetition-penalty", "1.1",
                   "--presence-penalty", "0.25", "--frequency-penalty", "0.5",
                   "--xtc-probability", "0.5", "--xtc-threshold", "0.1",
                   "--prefill-step-size", "512",
                   "--system-prompt", "sys"])
    assert rc == 0
    assert gen["stop"] == ["END", "DONE"]              # repeatable, in order
    assert gen["kv_bits"] == 8 and gen["kv_group_size"] == 32
    assert gen["max_kv_size"] == 4096 and gen["quantized_kv_start"] == 16
    assert gen["thinking_budget"] == 100
    assert gen["repetition_penalty"] == 1.1
    assert gen["presence_penalty"] == 0.25 and gen["frequency_penalty"] == 0.5
    assert gen["xtc_probability"] == 0.5 and gen["xtc_threshold"] == 0.1
    assert gen["prefill_step_size"] == 512
    assert gen["system_prompt"] == "sys"
    for k in ("kv_bits", "kv_group_size", "max_kv_size", "thinking_budget",
              "prefill_step_size"):
        assert isinstance(gen[k], int)                 # type=int survived


def test_run_no_chat_template_and_seed(gguf, gen, monkeypatch):
    import mlx.core as mx

    seeded = []
    monkeypatch.setattr(mx.random, "seed", lambda v: seeded.append(v))
    rc = cli.main([gguf, "--no-chat-template", "--seed", "7", "--prompt", "hi"])
    assert rc == 0
    assert gen["apply_chat_template"] is False
    assert gen["prompt"] == "hi"
    assert seeded == [7]                               # --seed reached mx.random.seed


# --prompt-file
def test_prompt_file_reads_prompt(gguf, gen, tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("prompt from file\n")
    assert cli.main([gguf, "--prompt-file", str(pf)]) == 0
    assert gen["prompt"] == "prompt from file\n"       # verbatim, newline kept


def test_prompt_file_missing_exits_2(gguf, gen, capsys):
    assert cli.main([gguf, "--prompt-file", "/no/such/prompt.txt"]) == 2
    err = capsys.readouterr().err
    assert "--prompt-file" in err and "error:" in err


# exit 1 = load refusal (docs/cli.md)
def test_run_codec_refusal_exits_1(gguf, monkeypatch, capsys):
    import gmlx.load.loader as loader
    from gmlx.load.preflight import UnsupportedCodecError

    exc = UnsupportedCodecError("gemma4", {"TQ1_0": 3})

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(loader, "load_model", boom)
    assert cli.main([gguf]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    # the refusal itself is one line naming the codec
    assert err.strip().splitlines()[-1] == f"error: {exc}"
    assert "TQ1_0" in str(exc)


# exit 2 = flag conflicts refused before any load. --speculative
# --stream-experts is NOT here anymore: streaming composes with MTP
# (placement after load_mtp_model; see resolve_speculative).
@pytest.mark.parametrize("extra,named", [
    (["--adapter", "/x.gguf", "--mmproj", "/mm.gguf"], "--mmproj"),
    (["--speculative", "--stream-cpu"], "--stream-cpu"),
])
def test_run_adapter_conflicts_exit_2(gguf, extra, named, capsys):
    assert cli.main([gguf, *extra]) == 2
    err = capsys.readouterr().err
    assert named in err and "not supported yet" in err


# --report-only
@pytest.fixture
def report_stubs(monkeypatch):
    """Stub the _report_only seams (wire bytes + remap + inventory + tokenizer);
    poison the model-build/generate seams. Returns the preflight module."""
    import gmlx.load.preflight as preflight_mod
    import gmlx.load.tokenizer as tok_mod
    import gmlx.gen.generation as generation
    import gmlx.load.loader as loader
    import gmlx.load.wire as wire

    monkeypatch.setattr(preflight_mod, "preflight", lambda path, arch=None: None)
    monkeypatch.setattr(
        wire, "load_gguf_wire_bytes",
        lambda path, zero_copy=True: ({"t": 0}, {"t": {}}, "gemma4", {}, {}))
    import gmlx.load.gguf_meta as gguf_meta
    monkeypatch.setattr(gguf_meta, "read_int", lambda meta, key: None)
    monkeypatch.setattr(gguf_meta, "first_nonzero_int", lambda meta, key: None)
    monkeypatch.setattr(
        wire, "remap_arrays",
        lambda arrays, kq, arch, no_remap=False, n_head=None, n_head_kv=None:
        (arrays, {}, {}))
    monkeypatch.setattr(loader, "print_inventory",
                        lambda *a, **k: print("[inventory] stub"))
    monkeypatch.setattr(loader, "_resolve_chat_template", lambda x: None)

    class _Tok:
        chat_template = "T"

        def apply_chat_template(self, msgs, tokenize=False,
                                add_generation_prompt=True):
            return "RENDERED: " + msgs[0]["content"]

    monkeypatch.setattr(tok_mod, "load_tokenizer_from_gguf",
                        lambda meta, arch, chat_template_override=None: _Tok())
    monkeypatch.setattr(
        loader, "load_model",
        lambda *a, **k: pytest.fail("--report-only must not build the model"))
    monkeypatch.setattr(
        generation, "generate",
        lambda *a, **k: pytest.fail("--report-only must not generate"))
    return preflight_mod


def test_report_only_inventories_and_skips_generate(gguf, report_stubs, capsys):
    assert cli.main([gguf, "--report-only", "--prompt", "hello"]) == 0
    out = capsys.readouterr().out
    assert "[inventory] stub" in out
    assert "=== rendered prompt ===" in out and "RENDERED: hello" in out


def test_report_only_swallows_arch_error(gguf, report_stubs, monkeypatch, capsys):
    # an arch the loader can't build must still inventory (rc 0)
    from gmlx.load.arch_table import UnsupportedArchError

    def boom(path, arch=None):
        raise UnsupportedArchError("arch 'foo' not supported")

    monkeypatch.setattr(report_stubs, "preflight", boom)
    assert cli.main([gguf, "--report-only"]) == 0
    assert "[inventory] stub" in capsys.readouterr().out


def test_report_only_codec_error_still_refuses(gguf, report_stubs, monkeypatch,
                                               capsys):
    from gmlx.load.preflight import UnsupportedCodecError

    def boom(path, arch=None):
        raise UnsupportedCodecError("gemma4", {"TQ1_0": 3})

    monkeypatch.setattr(report_stubs, "preflight", boom)
    assert cli.main([gguf, "--report-only"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "TQ1_0" in err


# --bench-depths list validation (before any load)
def test_bench_depths_bad_list_exits_2(gguf, monkeypatch, capsys):
    import gmlx.load.loader as loader
    import gmlx.spec.mtp_load as mtp_load

    monkeypatch.setattr(loader, "load_model",
                        lambda *a, **k: pytest.fail("must not load"))
    monkeypatch.setattr(mtp_load, "load_mtp_model",
                        lambda *a, **k: pytest.fail("must not load"))
    assert cli.main([gguf, "--bench-depths", "4096,abc"]) == 2
    err = capsys.readouterr().err
    assert "--bench-depths" in err and "not a comma-separated int list" in err


# --max-tokens: unset = generate until EOS (huge effective cap); N = cap
def test_max_tokens_default_uncapped(gguf, gen, capsys):
    assert cli.main([gguf]) == 0
    assert gen["max_tokens"] == cli._UNCAPPED_MAX_TOKENS
    assert "max_tokens=until-eos" in capsys.readouterr().out


def test_max_tokens_explicit_cap(gguf, gen, capsys):
    assert cli.main([gguf, "--max-tokens", "64"]) == 0
    assert gen["max_tokens"] == 64
    assert "max_tokens=64" in capsys.readouterr().out


def test_warn_cap_hit_only_when_capped(capsys):
    class A:
        pass

    a = A()
    a.max_tokens = 64
    a._max_tokens_capped = True
    cli.warn_cap_hit(a, 64)
    assert "--max-tokens cap (64)" in capsys.readouterr().err
    a.max_tokens = cli._UNCAPPED_MAX_TOKENS
    a._max_tokens_capped = False
    cli.warn_cap_hit(a, 10)
    assert capsys.readouterr().err == ""


def test_run_reasoning_default_and_explicit(gguf, gen):
    assert cli.main([gguf]) == 0
    assert gen["reasoning"] == "show"          # default matches chat
    assert cli.main([gguf, "--reasoning", "raw"]) == 0
    assert gen["reasoning"] == "raw"


def test_vlm_path_keeps_fp16_under_kvarn(capsys):
    # The VLM path has no kvarn cache, so kvarn keeps KV fp16 there rather
    # than passing the width on to an affine cache.
    import types
    import gmlx.commands.cli as cli

    assert cli.vlm_declines_kvarn(types.SimpleNamespace(kv_quant_scheme="kvarn"))
    assert "KV stays fp16" in capsys.readouterr().err
    assert not cli.vlm_declines_kvarn(
        types.SimpleNamespace(kv_quant_scheme="uniform"))
    assert not cli.vlm_declines_kvarn(types.SimpleNamespace())


def test_vlm_kv_width_installs_the_rotating_safe_quantizer(monkeypatch, capsys):
    # A gemma-4 stack has sliding-window layers. The stock mlx-vlm quantizer
    # calls to_quantized on every cache and raises "RotatingKVCache
    # Quantization NYI" on them, so an image turn with --kv-bits stopped at
    # the first prefill chunk. The VLM width helper installs the
    # rotating-safe quantizer serve uses before it hands the width on.
    import types

    import mlx.core as mx

    import gmlx.cache.apc_pooling as apc_pooling
    import gmlx.commands.cli as cli
    from gmlx._vendor.mlx_vlm.generate import common
    from gmlx._vendor.mlx_vlm.generate.ar import _generate_module_override
    from gmlx.cache.kv_cache import KVCache, RotatingKVCache

    installs = []
    real = apc_pooling.install_safe_kv_quantization

    def recording_install():
        installs.append(True)
        real()

    monkeypatch.setattr(
        apc_pooling, "install_safe_kv_quantization", recording_install)
    model = types.SimpleNamespace(layers=[])
    args = types.SimpleNamespace(kv_bits=8, kv_quant_scheme=None)

    assert cli.vlm_kv_width(args, model) == 8
    assert installs == [True]

    mx.random.seed(5)
    rot = RotatingKVCache(max_size=4)
    for _ in range(6):  # rotated: the stock to_quantized raises NYI
        rot.update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0)))
    kv = KVCache()
    kv.update_and_fetch(
        mx.random.normal((1, 1, 3, 32)), mx.random.normal((1, 1, 3, 32)))
    pc = [rot, kv]
    # The generate loop resolves the quantizer from the mlx_vlm.generate
    # package at call time, which is where the installer publishes it.
    quantize = _generate_module_override(
        "maybe_quantize_kv_cache", common.maybe_quantize_kv_cache)
    quantize(pc, quantized_kv_start=0, kv_group_size=32, kv_bits=8,
             kv_quant_scheme=None, kv_key_bits=None, kv_value_bits=None)
    assert pc[0] is rot  # the window layer stays fp16
    assert type(pc[1]).__name__ == "QuantizedKVCache"  # the full layer packs

    # No width, or a decline, hands no width on and installs nothing more.
    assert cli.vlm_kv_width(
        types.SimpleNamespace(kv_bits=None, kv_quant_scheme=None), model) is None
    assert cli.vlm_kv_width(
        types.SimpleNamespace(kv_bits=8, kv_quant_scheme="kvarn"), model) is None
    assert "KV stays fp16" in capsys.readouterr().err
    assert installs == [True]


def test_auto_scheme_normalizes_to_unset():
    # Every cache builder picks per model on an unset scheme; the image
    # path hands the value to mlx-vlm, which has no "auto".
    import types
    import gmlx.commands.cli as cli

    args = types.SimpleNamespace(kv_quant_scheme="auto")
    cli.normalize_kv_scheme(args)
    assert args.kv_quant_scheme is None
    args = types.SimpleNamespace(kv_quant_scheme="kvarn")
    cli.normalize_kv_scheme(args)
    assert args.kv_quant_scheme == "kvarn"


# --stream-experts streams a model that fits too.

def _small_moe(monkeypatch, budget_bytes):
    """A one-layer MoE under (or over) a fake stream budget, with the
    process-wide wired-limit patches the install makes undone at teardown."""
    import importlib
    import types

    import mlx.core as mx

    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    for name in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils"):
        mod = importlib.import_module(name)
        if hasattr(mod, "wired_limit"):
            monkeypatch.setattr(mod, "wired_limit", mod.wired_limit)
    monkeypatch.setattr(
        mx, "device_info",
        lambda *a: {"max_recommended_working_set_size": budget_bytes / 0.9})
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    layer = types.SimpleNamespace(modules=lambda: [glu])
    model = types.SimpleNamespace(layers=[layer], parameters=lambda: {"glu": glu.parameters()})
    return model, glu


def test_stream_experts_streams_a_model_under_the_budget(monkeypatch):
    import types

    import mlx.core as mx

    from gmlx.upstream.patching import wrapped_with

    model, glu = _small_moe(monkeypatch, 1 << 40)
    cli._apply_placement(types.SimpleNamespace(stream_experts=True, gguf=None), model)
    assert glu.__class__.__name__.endswith("_CPUOffload")       # wrapped
    assert getattr(glu, "_kq_cpu_only", False)                  # experts stream
    assert model._kq_streaming is True
    assert wrapped_with(mx.set_wired_limit, "_kq_no_sweep")     # sweep neutralized


def test_stream_flags_price_the_prefill_step_the_run_asks_for(monkeypatch):
    """The KV room prices the prefill chunk at --prefill-step-size, which
    run and chat pass to generation as an argument, not through the
    environment serve exports."""
    import types

    import gmlx.stream.expert_streaming as es

    seen = []

    def install(model, **kw):
        seen.append(kw.get("prefill_step"))
        return 1, 0

    monkeypatch.setattr(es, "install_expert_streaming", install)
    monkeypatch.setattr(es, "configure_stream_cpu", install)
    for flag in ("stream_experts", "stream_cpu"):
        cli._apply_placement(types.SimpleNamespace(
            **{flag: True}, gguf=None, prefill_step_size=16384), object())
    cli._apply_placement(types.SimpleNamespace(stream_experts=True, gguf=None),
                         object())
    assert seen == [16384, 16384, None]


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("tables,streams", [(0, False),
                                           (28_800_000_000, False),
                                           (28_800_000_000, True)])
def test_without_the_flag_tables_that_bring_the_model_under_install_once(
        monkeypatch, tables, streams, mtp):
    """No stream flag, on an arch that gathers its tables per layer: tables
    that alone bring the model under the budget serve from the file. An
    MTP load installs before its wire, and the placement after it is then
    a no-op. Experts that do not fit the memory free now stream, and no
    lift of the command-buffer caps follows the load."""
    import types

    import gmlx.serve.capacity as cap
    import gmlx.stream.expert_streaming as es

    seen, lifted = [], []

    def install(model, **kw):
        seen.append(kw.get("gguf_path"))
        model._kq_streaming = streams
        return 1, 0

    monkeypatch.setattr(es, "install_expert_streaming", install)
    monkeypatch.setattr(cap, "gathered_table_bytes",
                        lambda p, *a, **k: tables)
    monkeypatch.setattr(cli, "_lift_live_cb_caps", lambda: lifted.append(1))
    model = types.SimpleNamespace()
    args = types.SimpleNamespace(stream_experts=False, stream_cpu=False,
                                 gguf="Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf")
    if mtp:
        es.install_unflagged_tables(model, args.gguf)  # the MTP load's
    cli._apply_placement(args, model)
    assert seen == (["Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf"] if tables else [])
    assert lifted == []


def test_without_the_flag_a_model_under_the_budget_stays_resident(monkeypatch):
    from gmlx.stream.expert_streaming import install_expert_streaming

    model, glu = _small_moe(monkeypatch, 1 << 40)
    n, _ = install_expert_streaming(model)
    assert n == 1
    assert not getattr(glu, "_kq_cpu_only", False)
    assert model._kq_streaming is False


def test_a_forced_stream_under_the_budget_frees_the_expert_headroom(monkeypatch):
    """The experts of a forced stream are page cache, as over the budget: the
    headroom estimate drops them, so it does not read them twice."""
    from mlx.utils import tree_flatten

    from gmlx.stream.expert_streaming import install_expert_streaming

    import gmlx.gen.prefill_decay as pd

    for name in ("_UNTRACKED_WEIGHTS", "_UNTRACKED_OWNERS", "_STREAMED_TRACKED",
                 "_STREAMED_CAP", "_BUILD_CREDIT"):
        monkeypatch.setattr(pd, name, {})
    model, glu = _small_moe(monkeypatch, 1 << 40)
    model._kq_weights_key = ("/models/moe.gguf",)
    pd.note_untracked_weights(0.0, key=model._kq_weights_key)   # all tracked
    _, offloaded = install_expert_streaming(model, explicit=True)
    assert offloaded == sum(a.nbytes for _, a in tree_flatten(glu.parameters()))
    assert pd.streamed_tracked_bytes() == offloaded


def _caps_device(monkeypatch, init):
    """cb_phase on a fake device that holds ``init``, with no caps in the
    env and no lift, and the prefill decay records a forced stream writes
    reset. Returns the device's caps, ``init`` first."""
    import types

    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.cb_phase as cb_phase

    for name in ("_UNTRACKED_WEIGHTS", "_UNTRACKED_OWNERS", "_STREAMED_TRACKED",
                 "_STREAMED_CAP", "_BUILD_CREDIT"):
        monkeypatch.setattr(pd, name, {})
    device = [init]

    def set_cb_caps(ops, mb):
        device.append((ops, mb))
        return device[-2]

    monkeypatch.setitem(cb_phase._state, "kq",
                        types.SimpleNamespace(set_cb_caps=set_cb_caps))
    for key, value in (("phase", None), ("caps", None), ("lifted", False),
                       ("pre_lift", None)):
        monkeypatch.setitem(cb_phase._state, key, value)
    for key in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
                "GMLX_CB_PHASE"):
        # setenv first, so teardown also drops a key the test sets.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    return device


def test_cb_phase_0_with_stream_experts_ends_the_lift_with_the_load(monkeypatch):
    """GMLX_CB_PHASE=0 with --stream-experts: the argv lift holds the load
    only. After the streamed install the device is back on the caps from
    before the lift, and no prefill or decode runs on (400, 100000)."""
    import os

    import gmlx.serve.cb_phase as cb_phase
    from gmlx.stream.expert_streaming import install_expert_streaming

    # The device latches the env lift at init.
    device = _caps_device(monkeypatch, cb_phase.COARSE)
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    cli._lift_stream_cb_caps(["run", "m.gguf", "--stream-experts"])
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "400"
    model, _ = _small_moe(monkeypatch, 1 << 40)
    install_expert_streaming(model, explicit=True)
    assert model._kq_streaming is True
    cb_phase.flip("prefill", model)
    cb_phase.flip("decode", model)
    assert device == [cb_phase.COARSE, cb_phase.FINE]


def test_a_config_stream_lift_ends_on_the_caps_found_before_it(monkeypatch):
    """A config placement lifts the live device for its load, after the
    device latched its own caps. The streamed install sets those back,
    here (40, 40), the device defaults of a Pro chip."""
    import gmlx.serve.cb_phase as cb_phase
    from gmlx.stream.expert_streaming import install_expert_streaming

    device = _caps_device(monkeypatch, (40, 40))
    cli._lift_live_cb_caps()
    model, _ = _small_moe(monkeypatch, 1 << 40)
    install_expert_streaming(model, explicit=True)
    assert device == [(40, 40), cb_phase.COARSE, (40, 40)]


def test_a_forced_stream_under_the_budget_sizes_a_half_arena(monkeypatch):
    import types

    import gmlx.gen.prefill_decay as decay
    import gmlx.stream.budget as sbudget
    import gmlx.stream.pin_weights as pins
    import gmlx.stream.prefetch as prefetch
    from gmlx.stream.expert_streaming import install_expert_streaming

    model, _ = _small_moe(monkeypatch, 1 << 40)
    monkeypatch.setattr(prefetch, "maybe_make_prefetcher",
                        lambda path: types.SimpleNamespace(offsets={}))
    monkeypatch.setattr(pins, "cast_copies", lambda m, p: types.SimpleNamespace(
        dead_bytes=0, names=frozenset()))
    monkeypatch.setattr(pins, "maybe_pin_weights", lambda *a, **kw: None)
    monkeypatch.setattr(sbudget, "kv_room_bytes",
                        lambda p, **kw: types.SimpleNamespace(bytes=0))
    monkeypatch.setattr(decay, "note_streamed_tracked_bytes", lambda *a, **kw: None)
    monkeypatch.setattr("gmlx.gen.prefill_decay.deduct_untracked_weights",
                        lambda n, key=None: None)
    seen = []

    class Reached(Exception):
        pass

    def arena(*a, **kw):
        seen.append(kw.get("forced"))
        raise Reached

    monkeypatch.setattr("gmlx.stream.expert_streaming._decode_arena_bytes", arena)
    with pytest.raises(Reached):
        install_expert_streaming(model, gguf_path="m.gguf", explicit=True,
                                 feeder_prefill=False, feeder_decode=False)
    assert seen == [True]


def test_both_stream_flags_latch_the_warm_touch_for_run_and_chat():
    """`run` and `chat` both call _ensure_stream_cb_caps before the load."""
    import types

    import gmlx.stream.table_stream as ts

    for flags in ({"stream_experts": True}, {"stream_cpu": True}):
        cli._ensure_stream_cb_caps(types.SimpleNamespace(**flags))
        assert ts._FORCED
        cli._ensure_stream_cb_caps(types.SimpleNamespace())
        assert not ts._FORCED


@pytest.fixture
def vlm_gen(monkeypatch, tmp_path):
    """Stub the VLM load and generate seams; the text tower streams its
    experts once placed. Returns the recorded generate kwargs."""
    import types

    import gmlx._vendor.mlx_vlm as vlm_pkg
    import gmlx._vendor.mlx_vlm.prompt_utils as prompt_utils
    import gmlx.load.vlm as vlm

    seen = {}
    tower = types.SimpleNamespace(model_type="deepseek_v4")
    model = types.SimpleNamespace(language_model=tower)
    monkeypatch.setattr(vlm, "load_vlm_model",
                        lambda *a, **k: (model, {}, object(), None))
    monkeypatch.setattr(cli, "_apply_placement",
                        lambda args, m: setattr(m, "_kq_streaming",
                                                bool(args.stream_experts)))
    monkeypatch.setattr(cli, "print_family_note", lambda args: None)
    monkeypatch.setattr(prompt_utils, "apply_chat_template",
                        lambda *a, **k: "prompt")
    monkeypatch.setattr(vlm_pkg, "generate",
                        lambda *a, **k: seen.update(k))
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"GGUF")
    return seen, str(mmproj)


@pytest.mark.parametrize("extra, step", [
    (["--stream-experts"], "streamed"),
    (["--stream-experts", "--prefill-step-size", "1024"], 1024),
    ([], None),
])
def test_vlm_run_takes_the_streamed_prefill_chunk(gguf, vlm_gen, extra, step):
    """A VLM whose text tower streams its experts prefills in the streamed
    chunk, as the text path does, unless --prefill-step-size sets one."""
    from gmlx.load import loader

    seen, mmproj = vlm_gen
    assert cli.main([gguf, "--mmproj", mmproj, "--prompt", "hi", *extra]) == 0
    want = (loader._STREAMING_PREFILL_STEP_BY_MODEL_TYPE.get(
        "deepseek_v4", loader._STREAMING_PREFILL_STEP)
        if step == "streamed" else step)
    assert seen.get("prefill_step_size") == want


# --mmproj: what mlx-vlm's budget criteria gets

_M3_TEMPLATE = ("{%- set think_begin_token = '<mm:think>' -%}"
                "{%- set think_end_token = '</mm:think>' -%}")
_M3_PAIR = {"thinking_start_token": "<mm:think>",
            "thinking_end_token": "</mm:think>"}


class _Proc:
    def __init__(self, template):
        self.tokenizer = type("Tok", (), {"chat_template": template})()


def test_vlm_budget_takes_the_template_markers_and_enables_the_criteria():
    # MiniMax-M3 --thinking off sets budget 0. The criteria acts only with
    # enable_thinking, and on <think> by default, so the model reasoned.
    from gmlx.gen.thinking_budget import vlm_budget_kwargs

    got = vlm_budget_kwargs(0, None, None, _Proc(_M3_TEMPLATE))
    assert got == {"enable_thinking": True, **_M3_PAIR}


def test_vlm_budget_kwargs_without_a_budget_or_known_markers():
    from gmlx.gen.thinking_budget import vlm_budget_kwargs

    proc = _Proc(_M3_TEMPLATE)
    assert vlm_budget_kwargs(None, None, None, proc) == {}
    assert vlm_budget_kwargs(None, "<a>", "</a>", proc) == {
        "thinking_start_token": "<a>", "thinking_end_token": "</a>"}
    # gemma-4's channel tags are no known pair: stay unarmed.
    assert vlm_budget_kwargs(8, None, None, _Proc("<|channel>thought")) == {}


def test_vlm_run_hands_the_budget_kwargs_to_generate(gguf, vlm_gen, monkeypatch):
    import types

    import gmlx.load.vlm as vlm

    seen, mmproj = vlm_gen
    tower = types.SimpleNamespace(model_type="minimax_m3")
    model = types.SimpleNamespace(language_model=tower)
    monkeypatch.setattr(vlm, "load_vlm_model", lambda *a, **k: (
        model, {}, _Proc(_M3_TEMPLATE), None))
    assert cli.main([gguf, "--mmproj", mmproj, "--prompt", "hi",
                     "--thinking-budget", "0"]) == 0
    assert seen["thinking_budget"] == 0
    assert seen["enable_thinking"] is True
    assert {k: seen[k] for k in _M3_PAIR} == _M3_PAIR
