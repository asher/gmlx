"""A streamable table the wire load maps outside MLX's tracker: MLX never
counts or wires it, and its install leaves the wired limit free. Only a
load whose caller installs the table stream maps it that way; any other
loads it tracked. An mlx-kquant that cannot map it that way loads it
tracked, and the install pins the limit as for any tracked table. The
first map puts the process under the wired cap; a process that maps
none keeps MLX's limits."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

import gmlx.serve.capacity as capacity
import gmlx.stream.table_stream as ts
import gmlx.stream.wired_limit as wired_limit
from gmlx.upstream.patching import wrapped_with
from helpers import _real_apple_gpu

gguf = pytest.importorskip("gguf")

TABLE = "per_layer_token_embd.weight"
ROWS, DIMS = 4096, 512


class _Node:
    pass


@pytest.fixture(autouse=True)
def _fresh_tables(monkeypatch):
    monkeypatch.setattr(ts, "_UNTRACKED", {})
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.delenv("GMLX_STREAM_PLE_COMPOSE", raising=False)
    # The stream flag latch: the load streams the table for sure.
    ts.force_table_stream(True)
    yield
    ts.force_table_stream(False)


@pytest.fixture(autouse=True)
def _own_limits(monkeypatch):
    """The wired cap a map installs goes when the test ends: the clamp on
    mx.set_wired_limit, the MLX memory limit and the latch."""
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    monkeypatch.setattr(capacity, "_CAP_ENGAGED", False)
    monkeypatch.setattr(capacity, "_CAP_PLANNED", set())
    monkeypatch.setattr(wired_limit, "_CAP_MEMORY_LIMIT", None)
    limit = mx.get_memory_limit()
    yield
    mx.set_memory_limit(limit)


@pytest.fixture
def flash_next_gguf(tmp_path):
    """A qwen4exp GGUF whose PLE table is iq4_nl, as in Flash-Next, at
    4096 rows of 512 weights (1.2 MB). Returns its path and wire bytes."""
    rng = np.random.default_rng(5)
    blocks = DIMS // 32
    wire = np.zeros((ROWS, blocks, 18), dtype=np.uint8)
    wire[:, :, 1] = 0x3C          # fp16 1.0 scale, little-endian
    wire[:, :, 2:] = rng.integers(0, 256, (ROWS, blocks, 16), dtype=np.uint8)
    wire = wire.reshape(ROWS, blocks * 18)
    path = str(tmp_path / "Flash-Next-Q4.gguf")
    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_tensor(TABLE, wire, raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, wire


def _model(arrays):
    """A qwen4exp model shell holding the loaded table."""
    from mlx_kquant.nn import KQuantEmbedding

    emb = KQuantEmbedding(ROWS, DIMS, "iq4_nl")
    emb.weight = arrays[TABLE]
    emb.scales = arrays["per_layer_token_embd.scales"]
    model = _Node()
    model.model_type = "qwen4_exp"
    model.model = _Node()
    model.model.ple_embed = emb
    return model


def _settled():
    mx.synchronize()
    return mx.get_active_memory()


@pytest.mark.skipif(not ts.untracked_tables_supported(),
                    reason="mlx-kquant has no load_gguf(untracked=...)")
@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="untracked views need the Metal backend")
def test_a_streamed_table_loads_untracked_and_leaves_the_limit_free(
        flash_next_gguf, monkeypatch):
    """The load maps the table outside MLX's tracker: active memory does
    not grow by it. The loader flags it, its install pins nothing, and
    rows gathered on the CPU stream match a tracked copy."""
    import mlx_kquant as kq

    from gmlx.load.wire import load_gguf_wire_bytes

    path, wire = flash_next_gguf
    ts.table_install_follows(path)
    ref = _model(kq.load_gguf(path, zero_copy=False)[0])
    mx.eval(ref.model.ple_embed.parameters())
    before = _settled()
    arrays = load_gguf_wire_bytes(path, zero_copy=True, shards=[path])[0]
    mx.eval(arrays[TABLE])
    assert _settled() - before < wire.nbytes // 4
    model = _model(arrays)
    assert ts.mark_untracked_tables(model, [path]) == wire.nbytes
    # The map put the process under the wired cap.
    assert capacity.cap_applies()
    assert wrapped_with(mx.set_wired_limit, "_kq_wired_cap")
    limit = mx.set_wired_limit
    ts.install_table_streaming(model)
    assert not wrapped_with(mx.set_wired_limit, "_kq_no_sweep")
    assert mx.set_wired_limit is limit
    ids = mx.array([[0, 7, 2048], [4095, 7, 1]], dtype=mx.int32)
    got = model.model.ple_embed(ids)
    want = ref.model.ple_embed(ids)
    assert bool(mx.all(got == want))


def _spy_kq_load(monkeypatch, error=None) -> list:
    """Record the ``untracked`` names of each ``kq.load_gguf`` call of the
    wire load; raise ``error`` on a call that asks for any."""
    import gmlx.load.wire as wire_mod

    real = wire_mod.kq.load_gguf
    calls: list = []

    def kq_load(*args, **kwargs):
        calls.append(sorted(kwargs.get("untracked", ())))
        if "untracked" in kwargs and error is not None:
            raise error
        return real(*args)

    monkeypatch.setattr(wire_mod.kq, "load_gguf", kq_load)
    return calls


@pytest.mark.parametrize("error", [
    TypeError("load_gguf() got an unexpected keyword argument 'untracked'"),
    RuntimeError("untracked: no Metal buffer view for this tensor"),
])
def test_a_table_that_cannot_load_untracked_loads_tracked_and_pins(
        flash_next_gguf, monkeypatch, error):
    """An mlx-kquant without load_gguf(untracked=...), or one that cannot
    view the table that way, reloads the shard tracked. The loader flags
    nothing, the plan under the wired cap made for the load ends with it,
    and the install pins the wired limit."""
    from gmlx.load.wire import load_gguf_wire_bytes

    path, wire = flash_next_gguf
    ts.table_install_follows(path)
    capacity.plan_cap(path)
    calls = _spy_kq_load(monkeypatch, error)
    arrays = load_gguf_wire_bytes(path, zero_copy=True, shards=[path])[0]
    assert calls == [[TABLE], []]
    assert not capacity.cap_applies()
    model = _model(arrays)
    assert ts.mark_untracked_tables(model, [path]) == 0
    ts.install_table_streaming(model)
    assert wrapped_with(mx.set_wired_limit, "_kq_no_sweep")


def test_a_load_no_caller_follows_with_the_install_maps_the_table_tracked(
        flash_next_gguf, monkeypatch):
    """The table streams for sure, but only a caller that declared the
    table stream's install on the model (the serve build, and run, chat,
    capture and kld through their placement) gets it untracked. Any other
    load, as gmlx train, the distill teacher and student, and a Python
    load_model make, maps it tracked: unwrapped, an untracked table would
    be gathered whole on the GPU with nothing counting it. A declared load
    asks for it untracked."""
    from gmlx.load.wire import load_gguf_wire_bytes

    path, wire = flash_next_gguf
    calls = _spy_kq_load(monkeypatch)
    stock, limit = mx.set_wired_limit, mx.get_memory_limit()
    arrays = load_gguf_wire_bytes(path, zero_copy=True, shards=[path])[0]
    assert calls == [[]]
    assert not ts.tables_load_untracked(path)
    assert ts.mark_untracked_tables(_model(arrays), [path]) == 0
    # Nothing mapped untracked: MLX's limits, and no plan under the cap.
    assert not capacity.cap_applies()
    assert mx.set_wired_limit is stock
    assert mx.get_memory_limit() == limit
    ts.table_install_follows(path)
    load_gguf_wire_bytes(path, zero_copy=True, shards=[path])
    assert calls == [[], [TABLE]]
    assert capacity.cap_applies()
    ts.table_install_follows(path, False)
    load_gguf_wire_bytes(path, zero_copy=True, shards=[path])
    assert calls == [[], [TABLE], []]


class _Stop(Exception):
    """Raised at the tokenizer, after every step of the model load."""


class _Shell(nn.Module):
    """A qwen4exp model as the loader builds it, with only the table."""

    def __init__(self):
        super().__init__()
        self.model_type = "qwen4_exp"
        self.model = nn.Module()
        self.model.ple_embed = nn.Embedding(ROWS, DIMS)


def _load_to_the_install(path, monkeypatch, via):
    """Run the declared load of ``path`` through ``via``: the whole of
    ``load_model``, or the wire load, the remap and ``_install_and_load``
    as the VLM loader drives them. Returns the model the load built and
    the table array the map handed back. With an mlx-kquant that cannot
    map a table untracked, the map hands back a tracked array the load
    takes as untracked: the steps between are the same."""
    import functools

    import gmlx.load.loader as loader
    import gmlx.load.wire as wire_mod

    mapped = {}
    real = wire_mod.kq.load_gguf
    supported = ts.untracked_tables_supported()

    # Keeps the docstring the support check reads.
    @functools.wraps(real)
    def kq_load(*args, **kwargs):
        if not supported:
            kwargs.pop("untracked", None)
        out = real(*args, **kwargs)
        if TABLE in out[0]:
            mapped["table"] = out[0][TABLE]
        return out

    def stop(*_a, **_k):
        raise _Stop

    model = _Shell()
    monkeypatch.setattr(wire_mod.kq, "load_gguf", kq_load)
    monkeypatch.setattr("gmlx.load.config_synth.synthesize_config",
                        lambda *a, **k: {"model_type": "qwen4_exp"})
    monkeypatch.setattr(loader, "build_model",
                        lambda cfg, **k: (model, dict(cfg)))
    monkeypatch.setattr("gmlx.load.tokenizer.load_tokenizer_from_gguf", stop)
    ts.table_install_follows(path)
    if via == "load_model":
        with pytest.raises(_Stop):
            loader.load_model(path)
    else:
        arrays, codecs = wire_mod.load_gguf_wire_bytes(
            path, zero_copy=True, shards=[path])[:2]
        hf_weights, hf_codecs, _ = wire_mod.remap_arrays(
            arrays, codecs, "qwen4exp")
        del arrays
        loader._install_and_load(
            model, hf_weights, hf_codecs, log=lambda *a, **k: None,
            sanitize=False, source_key=loader.weights_source_key(path))
    return model, mapped["table"]


def _install_takes_the_map(model, mapped) -> None:
    """The install gets the very array the map handed back. A step that
    read the table into a copy of its own would hand the install a
    buffer MLX tracks and a raised wired limit wires whole, under the
    untracked flag the map left by name, so the install would not pin."""
    emb = model.model.ple_embed
    assert emb.weight is mapped, "the load copied the table before its install"
    assert emb._kq_table_untracked
    assert capacity.cap_applies()
    ts.install_table_streaming(model)
    assert not wrapped_with(mx.set_wired_limit, "_kq_no_sweep")


@pytest.mark.parametrize("via", ["load_model", "_install_and_load"])
def test_the_declared_load_hands_the_mapped_table_to_its_install(
        flash_next_gguf, monkeypatch, via):
    """From the map to the install, every step of the model load passes
    the table on as the array the map handed back."""
    path, _ = flash_next_gguf
    _install_takes_the_map(*_load_to_the_install(path, monkeypatch, via))


@pytest.mark.parametrize("via", ["load_model", "_install_and_load"])
def test_a_step_that_copies_the_table_fails_the_install_check(
        flash_next_gguf, monkeypatch, via):
    """The check catches a new step that reads the table into a copy
    before the install, planted here in the out-scale install."""
    import gmlx.load.loader as loader

    path, _ = flash_next_gguf
    real = loader.install_out_scales

    def copying(model, hf_weights, hf_kquant_meta=None):
        key = "model.ple_embed.weight"
        hf_weights[key] = mx.array(np.array(hf_weights[key]))
        return real(model, hf_weights, hf_kquant_meta)

    monkeypatch.setattr(loader, "install_out_scales", copying)
    model, mapped = _load_to_the_install(path, monkeypatch, via)
    with pytest.raises(AssertionError, match="copied the table"):
        _install_takes_the_map(model, mapped)


# The functions in gmlx that call table_install_follows, as (file,
# function), and those that call the CLI's declaration, each before a
# load whose model its file then places.
_DECLARERS = {("serve/residency_build.py", "_build_chat"),
              ("commands/cli.py", "_declare_table_install")}
_CLI_DECLARERS = {("commands/cli.py", "_main"), ("tui/chat.py", "cmd_chat"),
                  ("commands/capture.py", "cmd_capture"),
                  ("commands/kld.py", "_load")}


def _gmlx_files():
    import gmlx

    root = Path(gmlx.__file__).parent
    for f in sorted(root.rglob("*.py")):
        if "_vendor" not in f.parts:
            yield f.relative_to(root).as_posix(), ast.parse(f.read_text())


def _called(node) -> set:
    return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(node) if isinstance(c, ast.Call)}


def _callers_of(name: str) -> set:
    """(file under gmlx/, function) for each top-level function, or method
    of a top-level class, in gmlx that calls ``name``."""
    out = set()
    for rel, tree in _gmlx_files():
        for node in tree.body:
            fns = (node.body if isinstance(node, ast.ClassDef) else [node])
            for fn in fns:
                if (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and name in _called(fn)):
                    out.add((rel, fn.name))
    return out


def test_only_loads_the_table_install_follows_declare_it():
    """The declaration is made in two places: the serve build, whose
    bridge installs the stream placement on what it loads (serve and its
    batch path), and the CLI's declaration, which run, chat, capture and
    kld make before a load whose model their file then places, directly
    or through the CLI's stream setup. Every other entry point (train,
    the distill teacher and student, the Python load_model, the image,
    video, speech and drafter loads) never declares, so its table loads
    tracked
    (``test_a_load_no_caller_follows_with_the_install_maps_the_table_tracked``).
    That each declared load hands the install the table it mapped is
    ``test_the_declared_load_hands_the_mapped_table_to_its_install``. A
    new declaration must be named here."""
    assert _callers_of("table_install_follows") == _DECLARERS
    cli = (_callers_of("_declare_table_install")
           | _callers_of("_ensure_stream_cb_caps"))
    assert cli - {("commands/cli.py", "_ensure_stream_cb_caps")} \
        == _CLI_DECLARERS


def test_a_serve_build_credits_a_table_that_fell_back_to_tracked(
        flash_next_gguf, monkeypatch):
    """A serve build that expected its table untracked gave it no build
    credit. When the load falls back to a tracked buffer, the table is
    credited to the build's owner from there, so the headroom does not
    read it as live memory before its install credits it."""
    import gmlx.gen.prefill_decay as pd
    from gmlx.load.wire import load_gguf_wire_bytes

    path, wire = flash_next_gguf
    ts.table_install_follows(path)
    monkeypatch.setattr(ts, "untracked_tables_supported", lambda: True)
    monkeypatch.setattr(pd, "_WEIGHTS_OWNER", "entry")
    monkeypatch.setattr(pd, "_BUILD_CREDIT", {})
    _spy_kq_load(monkeypatch, RuntimeError("untracked: no buffer view"))
    load_gguf_wire_bytes(path, zero_copy=True, shards=[path])
    assert pd._BUILD_CREDIT["entry"][1] == wire.nbytes
