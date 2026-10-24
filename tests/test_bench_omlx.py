"""The oMLX arm of bench/serve-bench.py and bench/repack-rec-mlx.py."""

import dataclasses
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "bench"


def _load(name, path):
    mod_spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(mod_spec)
    # dataclasses look the module up by name while the script runs
    sys.modules[name] = mod
    try:
        mod_spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load("serve_bench", _BENCH / "serve-bench.py")


@pytest.fixture(scope="module")
def repack():
    return _load("repack_rec_mlx", _BENCH / "repack-rec-mlx.py")


def _write_safetensors(path, tensors):
    header, data = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _read_safetensors(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + n]), raw[8 + n:]


def test_repack_relabels_rec_weights_and_declares_the_8bit_embedding(repack, tmp_path):
    src, dst = tmp_path / "rec", tmp_path / "out"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(
        {"quantization": {"group_size": 64, "bits": 4}}))
    (src / "tokenizer.json").write_text("{}")
    emb = bytes(range(256)) * 2            # 4 rows x 128 bytes: 128 cols at 8 bits
    lin = bytes(range(255, -1, -1))        # 2 rows x 128 bytes: 256 cols at 4 bits
    scale = b"\0" * 16
    _write_safetensors(src / "model.safetensors", {
        "embed.weight": ("U8", [4, 128], emb),
        "embed.scales": ("F16", [4, 2], scale),
        "layer.weight": ("U8", [2, 128], lin),
        "layer.scales": ("F16", [2, 4], scale),
        "norm.weight": ("U8", [8], b"\1" * 8),
    })

    repack.main([str(src), str(dst)])

    header, data = _read_safetensors(dst / "model.safetensors")
    _, src_data = _read_safetensors(src / "model.safetensors")
    assert data == src_data
    assert header["embed.weight"]["dtype"] == "U32"
    assert header["embed.weight"]["shape"] == [4, 32]
    assert header["layer.weight"]["shape"] == [2, 32]
    assert header["norm.weight"]["dtype"] == "U8"     # no .scales: left alone
    quant = json.loads((dst / "config.json").read_text())["quantization"]
    assert quant == {"group_size": 64, "bits": 4,
                     "embed": {"group_size": 64, "bits": 8}}
    assert (dst / "tokenizer.json").samefile(src / "tokenizer.json")


def test_omlx_arm_serves_only_the_current_model_with_its_caches_off(bench, tmp_path):
    first, second = tmp_path / "ckpt-a", tmp_path / "ckpt-b"
    first.mkdir()
    second.mkdir()
    args = SimpleNamespace(outdir=str(tmp_path / "results"), dry_run=False,
                           omlx_python="python3")
    a = bench.ModelSpec(name="model-a", gguf="a.gguf", tokenizer="gguf",
                        omlx_model=str(first))
    b = bench.ModelSpec(name="model-b", gguf="b.gguf", tokenizer="gguf",
                        omlx_model=str(second))

    bench.omlx_cmd(args, a, None, 4096)
    saved = tmp_path / "results" / "omlx-base" / "model-b" / "settings.json"
    saved.parent.mkdir(parents=True)
    saved.write_text('{"max_concurrent_requests": 1}')   # an earlier run's omlx_args
    cmd = bench.omlx_cmd(args, b, None, 4096)

    models = Path(cmd[cmd.index("--model-dir") + 1])
    assert [p.name for p in models.iterdir()] == ["model-b"]
    assert (models / "model-b").resolve() == second.resolve()
    assert "--no-cache" in cmd
    assert Path(cmd[cmd.index("--base-path") + 1]) == saved.parent
    assert not saved.exists()
    assert bench.omlx_cmd(args, b, 3, 4096) is None   # no MTP arm on omlx


def _fake_omlx(monkeypatch, with_ignore_eos):
    fields = {"temperature": float}
    if with_ignore_eos:
        fields["ignore_eos"] = bool
    params = dataclasses.make_dataclass(
        "SamplingParams", [(k, v, dataclasses.field(default=v())) for k, v in fields.items()],
        namespace={"__post_init__": lambda self: None})
    pkg, request, cli = ModuleType("omlx"), ModuleType("omlx.request"), ModuleType("omlx.cli")
    request.SamplingParams = params
    cli.main = lambda: 0
    pkg.request, pkg.cli = request, cli
    for name, mod in (("omlx", pkg), ("omlx.request", request), ("omlx.cli", cli)):
        monkeypatch.setitem(sys.modules, name, mod)
    return params


def test_omlx_wrapper_forces_every_reply_to_max_tokens(monkeypatch):
    params = _fake_omlx(monkeypatch, with_ignore_eos=True)
    _load("omlx_serve", _BENCH / "omlx-serve.py")
    assert params(ignore_eos=False).ignore_eos is True


def test_omlx_wrapper_refuses_an_omlx_without_forced_length(monkeypatch):
    _fake_omlx(monkeypatch, with_ignore_eos=False)
    with pytest.raises(SystemExit, match="0.7.2"):
        _load("omlx_serve", _BENCH / "omlx-serve.py")
